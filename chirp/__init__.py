"""chirp – real-time bird feeder detection, tracking, and species counting."""
__version__ = "0.2.0"

from chirp.birdcounter import BirdCounter
from chirp.config import ChirpConfig, load_config
from chirp.database import ChirpDatabase

__all__ = ["BirdCounter", "ChirpConfig", "load_config", "ChirpDatabase"]
