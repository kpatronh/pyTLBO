import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        pass

    
    @staticmethod
    def scatter(x, y, title='Title', xaxis_title='x', yaxis_title='y'):
        plt.plot(x, y, '--bo')
        plt.xlabel(xaxis_title)
        plt.ylabel(yaxis_title)
        plt.title(title)
        plt.show()

    @staticmethod
    def scatter_interactive(x, y, title='Title', xaxis_title='x', yaxis_title='y'):
        trace = go.Scatter(x=x, y=y, mode='lines+markers')
        data = [trace]
        layout = dict(title=title,
                    xaxis=dict(title=xaxis_title),
                    yaxis=dict(title=yaxis_title))
        fig = dict(data=data, layout=layout)
        plot(fig, filename=title+'.html')




