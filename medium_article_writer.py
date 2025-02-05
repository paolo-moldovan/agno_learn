"""
Based on blog_post_generator.py from agno
"""
import json
from textwrap import dedent
from typing import Dict, Iterator, Optional

from agno.agent import Agent
# Replace the OpenAIChat import with the OllamaChat model.
from agno.models.ollama import Ollama
from agno.storage.workflow.sqlite import SqliteWorkflowStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.utils.log import logger
from agno.utils.pprint import pprint_run_response
from agno.workflow import RunEvent, RunResponse, Workflow
from pydantic import BaseModel, Field


class MediumArticle(BaseModel):
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(..., description="Summary of the article if available.")


class SearchResults(BaseModel):
    articles: list[MediumArticle]


class ScrapedArticle(BaseModel):
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(..., description="Summary of the article if available.")
    content: Optional[str] = Field(
        ...,
        description="Full article content in markdown format. None if content is unavailable.",
    )


class MediumArticleGenerator(Workflow):
    """Advanced workflow for generating professional Medium articles with proper research and citations.

    This workflow orchestrates multiple AI agents to research, analyze, and craft
    compelling Medium articles that combine journalistic rigor with engaging storytelling.
    The system excels at creating content that is both informative and optimized for
    the Medium platform.
    """

    # Use the OllamaChat model (here using "llama3" as an example) for all agents.
    searcher: Agent = Agent(
        model=Ollama(id="llama3.1"),
        tools=[DuckDuckGoTools()],
        description=dedent("""\
        You are MediumResearch-X, an elite research assistant specializing in discovering
        high-quality sources for compelling Medium content. Your expertise includes:

        - Finding authoritative and trending sources
        - Evaluating content credibility and relevance
        - Identifying diverse perspectives and expert opinions
        - Discovering unique angles and insights
        - Ensuring comprehensive topic coverage
        """),
        instructions=(
            "1. Search Strategy 🔍\n"
            "   - Find 10-15 relevant sources and select the 5-7 best ones\n"
            "   - Prioritize recent, authoritative content\n"
            "   - Look for unique angles and expert insights\n"
            "2. Source Evaluation 📊\n"
            "   - Verify source credibility and expertise\n"
            "   - Check publication dates for timeliness\n"
            "   - Assess content depth and uniqueness\n"
            "3. Diversity of Perspectives 🌐\n"
            "   - Include different viewpoints\n"
            "   - Gather both mainstream and expert opinions\n"
            "   - Find supporting data and statistics"
        ),
        response_model=SearchResults,
        structured_outputs=True,
    )

    article_scraper: Agent = Agent(
        model=Ollama(id="llama3.1"),
        tools=[Newspaper4kTools()],
        description=dedent("""\
        You are ContentBot-X, a specialist in extracting and processing digital content
        for article creation. Your expertise includes:

        - Efficient content extraction
        - Smart formatting and structuring
        - Key information identification
        - Quote and statistic preservation
        - Maintaining source attribution
        """),
        instructions=(
            "1. Content Extraction 📑\n"
            "   - Extract content from the article\n"
            "   - Preserve important quotes and statistics\n"
            "   - Maintain proper attribution\n"
            "   - Handle paywalls gracefully\n"
            "2. Content Processing 🔄\n"
            "   - Format text in clean markdown\n"
            "   - Preserve key information\n"
            "   - Structure content logically\n"
            "3. Quality Control ✅\n"
            "   - Verify content relevance\n"
            "   - Ensure accurate extraction\n"
            "   - Maintain readability"
        ),
        response_model=ScrapedArticle,
        structured_outputs=True,
    )

    writer: Agent = Agent(
        model=Ollama(id="llama3.1"),
        description=dedent("""\
        You are MediumMaster-X, an elite content creator blending journalistic excellence
        with the captivating style required for Medium articles. Your strengths include:

        - Crafting attention-grabbing headlines
        - Writing engaging introductions
        - Structuring content for digital consumption
        - Incorporating research seamlessly
        - Optimizing for SEO while maintaining quality
        - Creating compelling conclusions
        """),
        instructions=(
            "1. Content Strategy 📝\n"
            "   - Craft attention-grabbing headlines\n"
            "   - Write a captivating introduction\n"
            "   - Structure content for reader engagement\n"
            "   - Include relevant subheadings\n"
            "2. Writing Excellence ✍️\n"
            "   - Balance expertise with accessibility\n"
            "   - Use clear, engaging language\n"
            "   - Include pertinent examples\n"
            "   - Incorporate statistics naturally\n"
            "3. Source Integration 🔍\n"
            "   - Cite sources properly\n"
            "   - Include expert quotes\n"
            "   - Maintain factual accuracy\n"
            "4. Digital Optimization 💻\n"
            "   - Structure for readability on Medium\n"
            "   - Include calls to action and shareable insights"
        ),
        expected_output=dedent("""\
        # {Attention-Grabbing Headline}

        ## Introduction
        {Engaging introduction setting the scene and context for a Medium audience}

        ## In-Depth Analysis
        {Detailed insights, expert opinions, and supportive data}
        
        ## Conclusion
        {Summarize key takeaways and include a call to action}

        ## References
        {Properly attributed references with links}
        """),
        markdown=True,
    )

    def run(
        self,
        topic: str,
        use_search_cache: bool = True,
        use_scrape_cache: bool = True,
        use_cached_report: bool = True,
    ) -> Iterator[RunResponse]:
        logger.info(f"Generating a Medium article on: {topic}")

        if use_cached_report:
            cached_medium_article = self.get_cached_medium_article(topic)
            if cached_medium_article:
                yield RunResponse(
                    content=cached_medium_article, event=RunEvent.workflow_completed
                )
                return

        search_results: Optional[SearchResults] = self.get_search_results(
            topic, use_search_cache
        )
        if search_results is None or len(search_results.articles) == 0:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content=f"Sorry, could not find any articles on the topic: {topic}",
            )
            return

        scraped_articles: Dict[str, ScrapedArticle] = self.scrape_articles(
            topic, search_results, use_scrape_cache
        )

        yield from self.write_medium_article(topic, scraped_articles)

    def get_cached_medium_article(self, topic: str) -> Optional[str]:
        logger.info("Checking if cached Medium article exists")
        return self.session_state.get("medium_articles", {}).get(topic)

    def add_medium_article_to_cache(self, topic: str, medium_article: str):
        logger.info(f"Saving Medium article for topic: {topic}")
        self.session_state.setdefault("medium_articles", {})
        self.session_state["medium_articles"][topic] = medium_article
        self.write_to_storage()

    def get_cached_search_results(self, topic: str) -> Optional[SearchResults]:
        logger.info("Checking if cached search results exist")
        return self.session_state.get("search_results", {}).get(topic)

    def add_search_results_to_cache(self, topic: str, search_results: SearchResults):
        logger.info(f"Saving search results for topic: {topic}")
        self.session_state.setdefault("search_results", {})
        self.session_state["search_results"][topic] = search_results.model_dump()
        self.write_to_storage()

    def get_cached_scraped_articles(
        self, topic: str
    ) -> Optional[Dict[str, ScrapedArticle]]:
        logger.info("Checking if cached scraped articles exist")
        return self.session_state.get("scraped_articles", {}).get(topic)

    def add_scraped_articles_to_cache(
        self, topic: str, scraped_articles: Dict[str, ScrapedArticle]
    ):
        logger.info(f"Saving scraped articles for topic: {topic}")
        self.session_state.setdefault("scraped_articles", {})
        self.session_state["scraped_articles"][topic] = scraped_articles
        self.write_to_storage()

    def get_search_results(
        self, topic: str, use_search_cache: bool, num_attempts: int = 3
    ) -> Optional[SearchResults]:
        if use_search_cache:
            try:
                search_results_from_cache = self.get_cached_search_results(topic)
                if search_results_from_cache is not None:
                    search_results = SearchResults.model_validate(
                        search_results_from_cache
                    )
                    logger.info(
                        f"Found {len(search_results.articles)} articles in cache."
                    )
                    return search_results
            except Exception as e:
                logger.warning(f"Could not read search results from cache: {e}")

        for attempt in range(num_attempts):
            try:
                searcher_response: RunResponse = self.searcher.run(topic)
                if (
                    searcher_response is not None
                    and searcher_response.content is not None
                    and isinstance(searcher_response.content, SearchResults)
                ):
                    article_count = len(searcher_response.content.articles)
                    logger.info(
                        f"Found {article_count} articles on attempt {attempt + 1}"
                    )
                    self.add_search_results_to_cache(topic, searcher_response.content)
                    return searcher_response.content
                else:
                    logger.warning(
                        f"Attempt {attempt + 1}/{num_attempts} failed: Invalid response type"
                    )
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{num_attempts} failed: {str(e)}")

        logger.error(f"Failed to get search results after {num_attempts} attempts")
        return None

    def scrape_articles(
        self, topic: str, search_results: SearchResults, use_scrape_cache: bool
    ) -> Dict[str, ScrapedArticle]:
        scraped_articles: Dict[str, ScrapedArticle] = {}

        if use_scrape_cache:
            try:
                scraped_articles_from_cache = self.get_cached_scraped_articles(topic)
                if scraped_articles_from_cache is not None:
                    scraped_articles = scraped_articles_from_cache
                    logger.info(
                        f"Found {len(scraped_articles)} scraped articles in cache."
                    )
                    return scraped_articles
            except Exception as e:
                logger.warning(f"Could not read scraped articles from cache: {e}")

        for article in search_results.articles:
            if article.url in scraped_articles:
                logger.info(f"Found scraped article in cache: {article.url}")
                continue

            article_scraper_response: RunResponse = self.article_scraper.run(
                article.url
            )
            if (
                article_scraper_response is not None
                and article_scraper_response.content is not None
                and isinstance(article_scraper_response.content, ScrapedArticle)
            ):
                scraped_articles[article_scraper_response.content.url] = article_scraper_response.content
                logger.info(f"Scraped article: {article_scraper_response.content.url}")

        self.add_scraped_articles_to_cache(topic, scraped_articles)
        return scraped_articles

    def write_medium_article(
        self, topic: str, scraped_articles: Dict[str, ScrapedArticle]
    ) -> Iterator[RunResponse]:
        logger.info("Writing Medium article")
        writer_input = {
            "topic": topic,
            "articles": [v.model_dump() for v in scraped_articles.values()],
        }
        yield from self.writer.run(json.dumps(writer_input, indent=4), stream=True)
        self.add_medium_article_to_cache(topic, self.writer.run_response.content)


if __name__ == "__main__":
    import random
    from rich.prompt import Prompt

    example_prompts = [
        "Why Cats Secretly Run the Internet",
        "The Science Behind Why Pizza Tastes Better at 2 AM",
        "Time Travelers' Guide to Modern Social Media",
        "How Rubber Ducks Revolutionized Software Development",
        "The Secret Society of Office Plants: A Survival Guide",
        "Why Dogs Think We're Bad at Smelling Things",
        "The Underground Economy of Coffee Shop WiFi Passwords",
        "A Historical Analysis of Dad Jokes Through the Ages",
    ]

    topic = Prompt.ask(
        "[bold]Enter a Medium article topic[/bold] (or press Enter for a random example)\n✨",
        default=random.choice(example_prompts),
    )

    url_safe_topic = topic.lower().replace(" ", "-")

    generate_medium_article = MediumArticleGenerator(
        session_id=f"generate-medium-article-on-{url_safe_topic}",
        storage=SqliteWorkflowStorage(
            table_name="generate_medium_article_workflows",
            db_file="tmp/workflows.db",
        ),
    )

    medium_article: Iterator[RunResponse] = generate_medium_article.run(
        topic=topic,
        use_search_cache=True,
        use_scrape_cache=True,
        use_cached_report=True,
    )

    pprint_run_response(medium_article, markdown=True)