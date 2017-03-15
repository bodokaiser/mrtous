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
                {% for images in filenames %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td><img src="images/{{ images[0] }}"></td>
                        <td><img src="images/{{ images[1] }}"></td>
                        <td><img src="images/{{ images[2] }}"></td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </body>
    </html>
'''

def split(items, size):
    for i in range(0, len(items), size):
        yield items[i:i+size]

def main(args):
    app = flask.Flask('mrtous')

    @app.route('/')
    def index():
        is_image = lambda f: f.endswith('.png')

        return flask.render_template_string(TEMPLATE,
            filenames=split(list(filter(is_image, os.listdir(args.outdir))), 3))

    @app.route('/images/<filename>')
    def images(filename):

        return flask.send_from_directory(args.outdir, filename)


    app.run(port=args.port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, nargs='?', default=5000)
    parser.add_argument('--outdir', type=str, nargs='?', default='output')

    main(parser.parse_args())