build:
	docker build -f Dockerfile -t "realtime-emotion-monitor" .

shell: build
	docker run -it -e DISPLAY=unix$(DISPLAY) -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0 -v $(shell pwd)/:/workspace "realtime-emotion-monitor" bash
