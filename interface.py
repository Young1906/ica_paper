import io
import random
from flask import Flask, Response, render_template
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from main import main

app = Flask(__name__)

@app.route('/plot.png')
def plot_png():
    fig = create_figure()[0]
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    # fig = Figure()
    # axis = fig.add_subplot(1, 1, 1)
    # xs = range(100)
    # ys = [random.randint(1, 50) for x in xs]
    # axis.plot(xs, ys)
    return main()

@app.route("/test")
def test():
    return render_template("plot.html")


if __name__ == "__main__":
    app.run()