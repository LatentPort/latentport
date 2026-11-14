"""Compatibility entry point; the demo is shipped in the installed package."""
from latentport._demo.server import main

if __name__ == '__main__':
    raise SystemExit(main())
