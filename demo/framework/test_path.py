import aecar

# test each of the built-in paths
for path in [aecar.Path.rose(), aecar.Path.lissajous(), aecar.Path.power_trig(), aecar.Path.line(), aecar.Path.parabola()]:
    path.plot_path()
    path.plot_time_data()