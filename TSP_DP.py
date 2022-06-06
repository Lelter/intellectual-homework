from concorde.tsp import TSPSolver
x, y = data["x"], data["y"]
# norm选择距离计算方式，可选的有 "EXPLICIT", "EUC_2D", "EUC_3D", "MAX_2D",
# "MAN_2D", "GEO", "GEOM", "ATT", "CEIL_2D", "DSJRAND"
solver = TSPSolver.from_data(xs=x, ys=y, norm="EUC_2D")