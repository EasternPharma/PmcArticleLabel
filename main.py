import time
import sys
from tqdm import tqdm
from CheckLibraries import main as check_libraries
from CheckVLLM import main as check_vllm
from ApiCall import ApiCall
from ArticleLabelHelper import ArticleLabelHelper

BATCH_SIZE = 100
POLL_INTERVAL_SECONDS = 5
API_BASE_URL  = "http://localhost:1368"
VLLM_BASE_URL = "http://localhost:8000"
MODEL_NAME    = "Qwen/Qwen3.5-4B"

api_client = ApiCall(base_url=API_BASE_URL)
label_helper = ArticleLabelHelper(
    vllm_base_url=VLLM_BASE_URL,
    model_name=MODEL_NAME,
)


def run_batch() -> bool:
    """Fetch one batch of unlabeled articles, label them, and push results to the server. Returns True if work was done."""
    articles = api_client.get_unlabeled_articles(batch_size=BATCH_SIZE)
    if not articles:
        return False

    results = label_helper.label_batch(articles)
    if not results:
        print("[main] No results produced from labeling.")
        return False

    success = api_client.update_article_labels(results)
    if success:
        print(f"[main] Updated {len(results)}/{len(articles)} articles.")
    return success


def main():
    """Entry point. Runs pre-flight checks, then continuously labels batches until interrupted with Ctrl+C."""
    if not check_libraries():
        sys.exit(1)
    if not check_vllm(vllm_base_url=VLLM_BASE_URL, model_name=MODEL_NAME):
        sys.exit(1)
    print("[main] Starting PMC article labeling loop.")
    total_batches = 0

    with tqdm(desc="Batches processed", unit="batch") as batch_bar:
        while True:
            try:
                had_work = run_batch()
                if had_work:
                    total_batches += 1
                    batch_bar.update(1)
                else:
                    batch_bar.set_postfix_str("waiting for articles...")
                    time.sleep(POLL_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                print(f"\n[main] Stopped. Total batches processed: {total_batches}")
                sys.exit(0)
            except Exception as e:
                print(f"[main] Unexpected error: {e}")
                time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
