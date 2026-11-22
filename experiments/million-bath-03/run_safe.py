"""Run the fingerprinted bath campaign with Windows checkpoint write retries."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from time import sleep


sys.dont_write_bytecode = True
spec = spec_from_file_location("bath_campaign", Path(__file__).with_name("run.py"))
campaign = module_from_spec(spec)
sys.modules[spec.name] = campaign
spec.loader.exec_module(campaign)
write_once = campaign.write_json


def write_with_retry(path, value):
    for attempt in range(8):
        try:
            return write_once(path, value)
        except PermissionError:
            if sys.platform != "win32" or attempt == 7:
                raise
            sleep(0.05 * (attempt + 1))


campaign.write_json = write_with_retry

if __name__ == "__main__":
    campaign.main()
