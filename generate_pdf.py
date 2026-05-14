from markdown_pdf import MarkdownPdf
from markdown_pdf import Section

pdf = MarkdownPdf(toc_level=2)
with open('project_overview_and_task_assignment.md', 'r', encoding='utf-8') as f:
    markdown_content = f.read()

pdf.add_section(Section(markdown_content))
pdf.save('project_overview_and_task_assignment.pdf')
print("PDF created successfully!")
