from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import os

STREAMLIT_URL = os.environ.get("STREAMLIT_APP_URL") or "https://culinary-agent-arinagarwal.streamlit.app/"


def main():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(STREAMLIT_URL)
        print(f"Opened {STREAMLIT_URL}")

        # Give the page time to fully load (Streamlit apps can be slow)
        time.sleep(10)

        print(f"Page title: {driver.title}")
        print(f"Current URL: {driver.current_url}")

        # Print all buttons found on page for debugging
        buttons = driver.find_elements(By.TAG_NAME, "button")
        print(f"Found {len(buttons)} button(s) on page:")
        for i, btn in enumerate(buttons):
            print(f"  Button {i}: text='{btn.text}', displayed={btn.is_displayed()}")

        # Try multiple selectors for the wake-up button
        wake_button = None

        # Method 1: exact text match
        xpaths = [
            "//button[contains(text(),'Yes, get this app back up')]",
            "//button[contains(text(),'get this app back up')]",
            "//button[contains(text(),'app back up')]",
            "//button[contains(.,'Yes, get this app back up')]",
            "//button[contains(.,'get this app back up')]",
        ]

        for xpath in xpaths:
            elements = driver.find_elements(By.XPATH, xpath)
            if elements:
                wake_button = elements[0]
                print(f"Found wake button with xpath: {xpath}")
                break

        if wake_button:
            print("Clicking wake-up button...")
            wake_button.click()
            time.sleep(5)
            print("Button clicked. App should be waking up.")
        else:
            # Check if the page shows sleeping indicators
            page_source = driver.page_source
            if "This app is" in page_source and "zzz" in page_source.lower():
                print("Page appears to show sleeping state but button not found.")
                print("Page source snippet:")
                print(page_source[:3000])
                exit(1)
            else:
                print("No wake-up button found. App is already awake.")

    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)
    finally:
        driver.quit()
        print("Script finished.")


if __name__ == "__main__":
    main()
