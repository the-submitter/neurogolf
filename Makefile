.PHONY: demo demo-live

demo:
	./scripts/launch_demo.sh

demo-live:
	./scripts/launch_demo.sh --live --tasks 11-12 --parallel 2
