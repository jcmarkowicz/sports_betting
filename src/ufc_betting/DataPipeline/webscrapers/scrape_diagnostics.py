"""Best-effort browser snapshots for diagnosing failed scraper lookups."""
import json
from pathlib import Path


def save_listing_diagnostics(driver, *, page_number, upcoming=False, directory="scraper_debug"):
    """Capture before navigation; diagnostic failures never mask the scrape error."""
    prefix = f"{'upcoming' if upcoming else 'completed'}_page_{page_number}"
    folder = Path(directory)
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"Could not create scraper diagnostics folder: {error}")
        return
    details = {}
    for name, read in (
        ("url", lambda: driver.current_url),
        ("title", lambda: driver.title),
        ("matching_rows", lambda: len(driver.find_elements("css selector", ".b-statistics__table-row"))),
    ):
        try:
            details[name] = read()
        except Exception as error:
            details[name] = f"Unavailable: {type(error).__name__}: {error}"
        print(f"Scraper diagnostic {name}: {details[name]}")
    for label, save in (
        ("metadata", lambda: (folder / f"{prefix}.json").write_text(json.dumps(details, indent=2), encoding="utf-8")),
        ("HTML", lambda: (folder / f"{prefix}.html").write_text(driver.page_source, encoding="utf-8")),
        ("screenshot", lambda: driver.save_screenshot(str(folder / f"{prefix}.png"))),
    ):
        try:
            result = save()
            if result is False:
                print(f"Could not save scraper {label}: browser returned False")
        except Exception as error:
            print(f"Could not save scraper {label}: {type(error).__name__}: {error}")
