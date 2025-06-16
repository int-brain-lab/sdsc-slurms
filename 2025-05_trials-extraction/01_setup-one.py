"""Set up ONE and download latest cache table."""
from pathlib import Path
from datetime import datetime, timedelta
from one.api import ONE

CACHE_DIR = Path.home().joinpath('Documents', 'PYTHON', 'trials-extraction', 'downloads')
one = ONE(base_url='https://alyx.internationalbrainlab.org', cache_dir=CACHE_DIR)
one.alyx.authenticate()
assert not one.offline and one.cache_dir == CACHE_DIR and CACHE_DIR.exists()
one.load_cache()
assert datetime.now() - one._cache._meta['created_time'] < timedelta(days=1)
# TODO Change default cache directory