import yt_dlp
import logging

logger = logging.getLogger(__name__)

_MAX_RESULTS = 5

# Silence yt-dlp's own console output
_YDL_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "extract_flat": True,   # don't download — metadata only
    "skip_download": True,
}


def _yt_search(query: str, max_results: int = _MAX_RESULTS) -> list[dict]:
    """
    Query YouTube directly via yt-dlp and return real watch-page URLs.
    Results come straight from YouTube — no hallucination possible.
    """
    search_query = f"ytsearch{max_results}:{query}"
    results = []
    try:
        with yt_dlp.YoutubeDL(_YDL_OPTS) as ydl:
            info = ydl.extract_info(search_query, download=False)

        for entry in (info.get("entries") or []):
            url   = entry.get("url") or entry.get("webpage_url") or ""
            title = (entry.get("title") or "").strip()
            if url and title:
                # Normalise to full watch URL if needed
                if not url.startswith("http"):
                    url = f"https://www.youtube.com/watch?v={url}"
                results.append({"title": title, "url": url})

    except Exception as e:
        logger.warning(f"yt-dlp search failed for '{query}': {e}")

    logger.info(f"YouTube search '{query}' → {len(results)} result(s)")
    return results


def search_youtube_for_topic(topic: str) -> list[dict]:
    """
    Search YouTube for tutorial videos on a CS topic.
    Called directly by EvaluationController — no LLM in the loop.

    Returns a list of dicts: [{title, url}, ...]
    """
    query = f"{topic} tutorial for beginners"
    return _yt_search(query)




