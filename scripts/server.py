import argparse
import flask
import os

TEMPLATE = '''
    <!DOCTYPE html>
    <html>
        <head>
            <title>mrtous</title>
        </head>
        <body>
            <canvas id="loss" width="400" height="100"></canvas>
            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Input</th>
                        <th>Output</th>
                        <th>Target</th>
                    </tr>
                </thead>
                <tbody>
                {% for image in images %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td><img src="images/{{ image[0] }}"></td>
                        <td><img src="images/{{ image[1] }}"></td>
                        <td><img src="images/{{ image[2] }}"></td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.5.0/Chart.min.js"></script>
            <script>
                var ctx = document.getElementById('loss')

                var events = {{ loss|safe }}
                var length = events.filter(event => event.epoch == 1).length
                var center = Math.ceil(length / 2)

                var chart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: events.map(event => [event.step, event.epoch]),
                        datasets: [{
                            label: 'Training',
                            fill: false,
                            backgroundColor: "rgba(75,192,192,0.4)",
                            borderColor: "rgba(75,192,192,1)",
                            data: events.map(event => event.loss)
                        }]
                    },
                    options: {
                        scales: {
                            xAxes: [
                                {
                                    id: 'steps',
                                    type: 'category',
                                    gridLines: {
                                        drawTicks: true
                                    },
                                    ticks: {
                                        callback: label => label.step
                                    }
                                },
                                {
                                    id: 'epochs',
                                    type: 'category',
                                    gridLines: {
                                        drawOnChartArea: false,
                                        drawTicks: false
                                    },
                                    ticks: {
                                        callback: label => {
                                            if (label[0] == center) {
                                                return label[1]
                                            }
                                            return ''
                                        }
                                    }
                                }
                            ]
                        }
                    }
                })
            </script>
        </body>
    </html>
'''

def is_image(filename):
    return filename.endswith('.png')

def split(items, size):
    for i in range(0, len(items), size):
        yield items[i:i+size]

def main(args):
    app = flask.Flask('mrtous')

    path = os.path.join(args.outdir, 'loss.json')

    @app.route('/')
    def index():
        if os.path.exists(path):
            with open(path, 'r') as f:
                loss = '[' + ','.join(f.read().split('\n'))[0:-1] + ']'
        else:
            loss = '[]'

        return flask.render_template_string(TEMPLATE, loss=loss,
            images=split(list(filter(is_image, os.listdir(args.outdir))), 3))

    @app.route('/images/<filename>')
    def images(filename):

        return flask.send_from_directory(args.outdir, filename)


    app.run(port=args.port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, nargs='?', default=5000)
    parser.add_argument('--outdir', type=str, nargs='?', default='output')

    main(parser.parse_args())