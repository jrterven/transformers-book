"""Export Chapter 1's executed notebook to written and code-and-output PDFs.

Install optional export dependencies with:
    python -m pip install reportlab mistune
Then run this file from any working directory. Execute the notebook first;
the exporter refuses missing outputs, errors, and out-of-order executions.
The written PDF includes every Markdown solution, excluding notebook setup.
The notebook PDF includes all Markdown, code cells, and recorded text outputs.
"""

from html import escape
import json
from pathlib import Path
import re
import textwrap

import mistune
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
WIDTH = letter[0] - 1.4 * inch
INK = colors.HexColor('#26323d')
BLUE = colors.HexColor('#3976af')
GRAY = colors.HexColor('#687580')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle('Body', fontName='Times-Roman', fontSize=10.5,
                          leading=14, spaceAfter=7, textColor=INK,
                          allowWidows=0, allowOrphans=0))
styles.add(ParagraphStyle('TitleBook', parent=styles['Body'],
                          fontName='Times-Bold', fontSize=22, leading=25,
                          spaceBefore=8, spaceAfter=12, keepWithNext=True))
styles.add(ParagraphStyle('Exercise', parent=styles['Body'],
                          fontName='Times-Bold', fontSize=15, leading=18,
                          spaceBefore=16, spaceAfter=8, keepWithNext=True))
styles.add(ParagraphStyle('TableBody', parent=styles['Body'],
                          fontName='Helvetica', fontSize=9, leading=11.5,
                          spaceAfter=0))
styles.add(ParagraphStyle('CellLabel', parent=styles['Body'],
                          fontName='Helvetica', fontSize=8, leading=11,
                          textColor=GRAY, spaceAfter=3, keepWithNext=True))
styles.add(ParagraphStyle('CodeBlock', fontName='Courier', fontSize=7.8, leading=10,
                          textColor=INK, backColor=colors.HexColor('#f3f6f8'),
                          borderPadding=7, spaceBefore=4, spaceAfter=10))
styles.add(ParagraphStyle('Output', parent=styles['CodeBlock'],
                          backColor=colors.HexColor('#eff6f2')))
markdown = mistune.create_markdown(renderer='ast', plugins=['table'])


def inline(nodes):
    result = []
    for node in nodes:
        kind = node['type']
        if kind == 'text':
            result.append(escape(node['raw']))
        elif kind in ('softbreak', 'linebreak'):
            result.append(' ' if kind == 'softbreak' else '<br/>')
        elif kind == 'codespan':
            result.append('<font name="Courier" size="9">' +
                          escape(node['raw']) + '</font>')
        elif kind in ('strong', 'emphasis'):
            tag = 'b' if kind == 'strong' else 'i'
            result.append(f'<{tag}>' + inline(node['children']) + f'</{tag}>')
        elif kind == 'link':
            result.append(inline(node['children']))
        else:
            raise ValueError(f'Unsupported inline Markdown: {kind}')
    return ''.join(result)


def fixed_width(source, style):
    lines = []
    for line in source.rstrip().splitlines():
        lines.extend(textwrap.wrap(line, width=96, replace_whitespace=False,
                                   drop_whitespace=False, subsequent_indent='    ') or [''])
    return Preformatted('\n'.join(lines), styles[style])


def table(token):
    head, body = token['children']
    rows = [head['children']] + [row['children'] for row in body['children']]
    raw = [[inline(cell['children']) for cell in row] for row in rows]
    weights = [max(8, min(38, max(len(re.sub('<[^>]+>', '', row[i]))
                                  for row in raw))) for i in range(len(raw[0]))]
    widths = [WIDTH * weight / sum(weights) for weight in weights]
    paragraphs = [[Paragraph(value, styles['TableBody']) for value in row]
                  for row in raw]
    result = Table(paragraphs, colWidths=widths, repeatRows=1, hAlign='LEFT')
    result.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef3')),
        ('LINEBELOW', (0, 0), (-1, 0), .6, BLUE),
        ('LINEBELOW', (0, -1), (-1, -1), .4, colors.HexColor('#bac6ce')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7f9fa')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return [KeepTogether([result, Spacer(1, 10)])]


def markdown_flowables(source):
    result = []
    for token in markdown(source):
        kind = token['type']
        if kind == 'blank_line':
            continue
        if kind == 'heading':
            style = 'TitleBook' if token['attrs']['level'] == 1 else 'Exercise'
            result.append(Paragraph(inline(token['children']), styles[style]))
        elif kind == 'paragraph':
            result.append(Paragraph(inline(token['children']), styles['Body']))
        elif kind == 'block_code':
            result.append(fixed_width(token['raw'], 'CodeBlock'))
        elif kind == 'table':
            result.extend(table(token))
        else:
            raise ValueError(f'Unsupported block Markdown: {kind}')
    return result


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#c5cfd7'))
    canvas.line(doc.leftMargin, letter[1]-40, letter[0]-doc.rightMargin, letter[1]-40)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(doc.leftMargin, letter[1]-32, 'TRANSFORMERS  |  CHAPTER 1')
    canvas.drawRightString(letter[0]-doc.rightMargin, letter[1]-32, doc.short_title)
    canvas.drawString(doc.leftMargin, 31, 'Tokens, Embeddings, and Neural Language Models')
    canvas.drawRightString(letter[0]-doc.rightMargin, 31, str(doc.page))
    canvas.restoreState()


def export(notebook, include_code):
    filename = 'python-solutions.pdf' if include_code else 'written-solutions.pdf'
    title = 'Worked solutions with Python' if include_code else 'Written solutions'
    story = []
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown':
            if not include_code and 'notebook-only' in cell.get('metadata', {}).get('tags', []):
                continue
            if source.startswith('# Chapter 1 | Worked solutions'):
                source = source.replace('# Chapter 1 | Worked solutions',
                                        '# Chapter 1 | ' + title, 1)
            if not include_code and re.match(r'## (3|5|7|9|11)\.', source):
                story.append(PageBreak())
            story.extend(markdown_flowables(source))
        elif cell['cell_type'] == 'code' and include_code:
            count = cell['execution_count']
            story.append(KeepTogether([
                Paragraph(f'In [{count}]', styles['CellLabel']),
                fixed_width(source, 'CodeBlock'),
            ]))
            for output in cell.get('outputs', []):
                if output['output_type'] == 'stream':
                    value = ''.join(output['text'])
                elif output['output_type'] in ('execute_result', 'display_data'):
                    value = ''.join(output['data']['text/plain'])
                else:
                    raise ValueError('Notebook has an unexportable output')
                story.append(KeepTogether([
                    Paragraph('Output', styles['CellLabel']), fixed_width(value, 'Output')]))
    doc = SimpleDocTemplate(str(ROOT / filename), pagesize=letter,
                            leftMargin=.7*inch, rightMargin=.7*inch,
                            topMargin=.75*inch, bottomMargin=.65*inch,
                            title='Chapter 1: ' + title,
                            author='Juan Terven and Diana Córdova')
    doc.short_title = title.upper()
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(filename, 'exported')


def main():
    notebook = json.loads((ROOT / 'python-solutions.ipynb').read_text())
    code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    counts = [cell.get('execution_count') for cell in code_cells]
    if counts != list(range(1, len(code_cells) + 1)):
        raise ValueError('Execute all notebook cells in order before exporting')
    if any(out.get('output_type') == 'error' for cell in code_cells
           for out in cell.get('outputs', [])):
        raise ValueError('Notebook contains an execution error')
    if any(not cell.get('outputs') for cell in code_cells[1:]):
        raise ValueError('An exercise is missing its executed output')
    export(notebook, include_code=False)
    export(notebook, include_code=True)


if __name__ == '__main__':
    main()
