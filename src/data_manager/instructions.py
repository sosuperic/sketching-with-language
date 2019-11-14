# instructions.py

"""
TODO: might refactor this into another file.
"""

import src.utils as utils
from src.models.instruction_gen import normalize

def convert_generated_instruction_samples_to_html(samples_fp):
    """
    Convert outputs from StrokeToInstructionRNN model to html
    """
    html_path = samples_fp.replace('.json', '.html')
    with open(html_path, 'w') as out_f:
        out_f.write("""
        <html lang="en">
            <head>
              <title>Bootstrap Example</title>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1">
              <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css">
              <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
              <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script>
            </head>
            <body>

            <div class="container">
                <h2>MTurk Results</h2>
        """)

        ROW_TEMPLATE = """
        <div class="row">
            <div class="col-md-5">
              <div class="thumbnail">
                  <div>
                   <p><strong>Category: {}</strong></p>
                  </div>
                  <img src="{}" style="max-width:100%">
                  <div class="caption">
                    <p>Ground truth: {}</p>
                    <p>Generated: {}
                  </div>
              </div>
            </div>
            <div class="col-md-5">
              <div class="thumbnail">
                  <div>
                   <p><strong>Category: {}</strong></p>
                  </div>
                  <img src="{}" style="max-width:100%">
                  <div class="caption">
                    <p>Ground truth: {}</p>
                    <p>Generated: {}
                  </div>
              </div>
            </div>
          </div>
        """

        samples = utils.load_file(samples_fp)
        for i in range(0, len(samples), 2):
            # cat = sample['category']
            cat1 = samples[i]['url'].split('fullinput/')[1].split('/progress')[0]
            url1 = samples[i]['url']
            gt1 = ' '.join(normalize(samples[i]['ground_truth']))
            gen1 = samples[i]['generated']

            cat2 = samples[i+1]['url'].split('fullinput/')[1].split('/progress')[0]
            url2 = samples[i+1]['url']
            gt2 = ' '.join(normalize(samples[i+1]['ground_truth']))
            gen2 = samples[i+1]['generated']

            row = ROW_TEMPLATE.format(cat1, url1, gt1, gen1,
                                      cat2, url2, gt2, gen2)
            out_f.write(row)

        out_f.write("""
            </div>
            </body>
        </html>
        """)


if __name__ == '__main__':
    samples_fp = 'samples_e11.json'
    convert_generated_instruction_samples_to_html(samples_fp)