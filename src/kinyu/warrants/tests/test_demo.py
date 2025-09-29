import pytest
from playwright.sync_api import Page, expect
import subprocess
import time
import os
import signal
import socket

def find_free_port():
    """Finds a free port on the host machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

@pytest.fixture(scope="module")
def live_server():
    """
    Fixture to start the provided `serve_demo.py` on a dynamic free port.
    This is the most robust approach as it uses the actual server script.
    """
    port = find_free_port()
    server_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    command = ["python3", "serve_demo.py", str(port)]

    # Start the server as a background process
    server_process = subprocess.Popen(
        command,
        cwd=server_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
        text=True
    )

    # Wait for the server to be ready
    time.sleep(2)

    # Check if the process has terminated prematurely
    if server_process.poll() is not None:
        stdout, stderr = server_process.communicate()
        print("\n--- SERVER FAILED TO START ---")
        print(f"Port used: {port}")
        print("--- STDOUT ---")
        print(stdout)
        print("--- STDERR ---")
        print(stderr)
        print("--------------------------\n")
        pytest.fail("Server process failed to start.", pytrace=False)

    print(f"Server started on http://localhost:{port} with PID {server_process.pid}")

    yield f"http://localhost:{port}"

    print(f"\nStopping server with PID {server_process.pid}...")
    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
    server_process.wait(timeout=5)
    print("Server stopped.")

def test_warrant_demo_calculation(live_server: str, page: Page):
    """
    Test the full user flow against the actual, dynamically-hosted server.
    """
    console_messages = []
    page.on("console", lambda msg: console_messages.append(f"[{msg.type}] {msg.text}"))

    try:
        demo_url = f"{live_server}/warrants_demo.html"
        page.goto(demo_url)
        expect(page).to_have_title("Exotic Warrant Pricing Demo")

        # Wait for the WASM module to load and enable the button
        calculate_button = page.locator("#calculateBtn")
        expect(calculate_button).to_be_enabled(timeout=20000) # Increased timeout for WASM loading

        page.locator("button.preset-btn:text-is('Low Risk')").click()
        calculate_button.click()

        loading_spinner = page.locator("#loading")
        expect(loading_spinner).to_be_hidden(timeout=60000)

        error_display = page.locator("#error")
        if error_display.is_visible():
            pytest.fail(f"UI Error: {error_display.inner_text()}", pytrace=False)

        result_display = page.locator("#result")
        expect(result_display).to_be_visible()

        price_display = page.locator("#priceDisplay")
        price_text = price_display.inner_text()

        # With a fixed seed, the price should be deterministic.
        # This assertion makes the test much more reliable.
        expected_price = "$14.168205"
        assert price_text == expected_price, f"Expected price {expected_price}, but got {price_text}"

        price_value = float(price_text.strip().replace("$", ""))
        assert price_value > 0, f"Calculated price was not positive: {price_value}"

        screenshot_path = "warrants_demo_result.png"
        page.screenshot(path=screenshot_path)
        print(f"\nScreenshot saved to: {screenshot_path}")
        assert os.path.exists(screenshot_path)

    except Exception as e:
        print("\n--- BROWSER CONSOLE LOGS ---")
        for msg in console_messages:
            print(msg)
        print("---------------------------\n")
        raise e