fps ?= 30
dir ?= out
out ?= vortex_street

render1:
	ffmpeg -framerate $(fps) -i $(dir)/vmag_fixed_%06d.png -c:v libx264 -crf 13 -tune animation -y $(out).mp4

render2:
	ffmpeg -framerate $(fps) -i $(dir)/vmag_fixed_%06d.png -framerate $(fps) -i $(dir)/vmag_comov_%06d.png -filter_complex "[1]pad=iw:ih+2:0:2[v1];[0][v1]vstack=inputs=2" -c:v libx264 -crf 13 -tune animation -y $(out).mp4

render3:
	ffmpeg -framerate $(fps) -i $(dir)/vmag_fixed_%06d.png -framerate $(fps) -i $(dir)/vmag_comov_%06d.png -framerate $(fps) -i $(dir)/curl_%06d.png -filter_complex "[1]pad=iw:ih+2:0:2[v1];[2]pad=iw:ih+2:0:2[v2];[0][v1][v2]vstack=inputs=3" -c:v libx264 -crf 13 -tune animation -y $(out).mp4
