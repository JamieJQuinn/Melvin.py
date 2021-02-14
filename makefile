png_files := $(patsubst %.npy,%.png,$(wildcard *.npy))
xav_files := $(patsubst %.npy,%_xav.png,$(wildcard *.npy))

plot: $(png_files) $(xav_files)

xav: $(xav_files)

slices: $(png_files)

%.png: %.npy
	pipenv run python plot.py $<

%_xav.png: %.npy
	pipenv run python plot_average.py $<
