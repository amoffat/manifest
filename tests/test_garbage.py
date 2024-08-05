import pytest

from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
import requests

DEBUG = False


garbage_check = pytest.mark.skipif(
    "not config.getoption('--check-garbage')",
    action="store_true",  # type: ignore
    default=False,  # type: ignore
    help="Only run when --check-garbage is given",  # type: ignore
)


@garbage_check
def test_gemini_not_garbage():
    url = "https://ai.google.dev/gemini-api/terms"
    response = requests.get(url)
    gemini_terms = response.text.lower()
    if DEBUG:
        print(gemini_terms)
    assert "develop models that compete" not in gemini_terms, "gemini is still garbage"


@garbage_check
def test_openai_not_garbage():
    url = "https://openai.com/policies/terms-of-use/"

    # Configure Chrome options for headless mode
    chrome_options = Options()
    # chrome_options.add_argument("--headless=new")

    # Set up the WebDriver service for improved compatibility
    service = Service("/usr/bin/chromedriver")  # Or your chromedriver path

    # Initialize the WebDriver
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Navigate to the URL and wait for the page to load (adjust timeout as needed)
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Get the entire body text for analysis
        openai_terms_element = driver.find_element(By.TAG_NAME, "body")
        openai_terms = openai_terms_element.text.lower()
        if DEBUG:
            print(openai_terms)
        # Assertions (add more checks as needed)
    finally:
        # Ensure the browser is closed even if there's an error
        driver.quit()
    assert "develop models that compete" not in openai_terms, "openai is still garbage"


@garbage_check
def test_anthropic_not_garbage():
    url = "https://www.anthropic.com/legal/consumer-terms"
    response = requests.get(url)
    anthropic_terms = response.text.lower()
    if DEBUG:
        print(anthropic_terms)
    assert (
        "develop any products or services that compete" not in anthropic_terms
    ), "anthropic is still garbage"


@garbage_check
def test_mistral_remote_not_garbage():
    url = "https://mistral.ai/terms/"
    response = requests.get(url)
    mistral_terms = response.text.lower()
    if DEBUG:
        print(mistral_terms)
    assert (
        "not use the services for a benefit of a third party" not in mistral_terms
    ), "mistral remote terms ... still garbage"


# closed, because "you may not use to benefit third party or whatever"
# def test_models_can_create_mistral_remote():
#     model_config = models.MistralRemoteModelConfig.OPEN_CODESTRAL_MAMBA
#     api_key = "FAKE"
#     llm = models.MistralRemote(
#         model_config=model_config,
#         api_key=api_key,
#     )
#     assert llm is not None
#     assert "Authorization" in llm.session.headers
#     assert llm.session.headers["Authorization"] == f"Bearer {api_key}"

# closed
# def test_models_can_create_anthropic():
#     model_config = models.AnthropicModelConfig.CLAUDE_3_5_SONNET
#     api_key = "FAKE"
#     llm = models.Anthropic(
#         model_config=model_config,
#         api_key=api_key,
#     )
#     assert llm is not None
#     assert "x-api-key" in llm.session.headers
#     assert llm.session.headers["x-api-key"] == api_key
# closed
# def test_models_can_create_gemini():
#     model_config = models.GeminiModelConfig.GEMINI_1_5_PRO
#     api_key = "FAKE"
#     llm = models.Anthropic(
#         model_config=model_config,
#         api_key=api_key,
#     )
#     assert llm is not None
#     assert "x-api-key" in llm.session.headers
#     assert llm.session.headers["x-api-key"] == api_key
# garbage
# def test_models_can_create_openai():
#     model_config = models.GeminiModelConfig.
#     api_key = "FAKE"
#     llm = models.Anthropic(
#         model_config=model_config,
#         api_key=api_key,
#     )
#     assert llm is not None
#     assert "x-api-key" in llm.session.headers
#     assert llm.session.headers["x-api-key"] == api_key
#     from random_user_agent.user_agent import UserAgent
#     from random_user_agent.params import SoftwareName, OperatingSystem
#     software_names = [
#         SoftwareName.CHROME.value,
#         SoftwareName.FIREFOX.value,
#         SoftwareName.OPERA.value,
#     ]
#     operating_systems = [
#         OperatingSystem.WINDOWS.value,
#         OperatingSystem.LINUX.value,
#         OperatingSystem.DARWIN.value,
#     ]
#     user_agent_rotator = UserAgent(
#         software_names=software_names,
#         operating_systems=operating_systems,
#         limit=100,
#     )
# user_agent = user_agent_rotator.get_random_user_agent()
# headers = {"user-agent": user_agent}
