import markdown
from xhtml2pdf import pisa
import os

# C'est ici que la magie opère : Le CSS définit la structure visuelle
CSS_STYLE = """
<style>
    @page {
        size: A4;
        margin: 2cm; /* Marges importantes pour la lisibilité */
    }
    body {
        font-family: Helvetica, Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.5; /* Espacement entre les lignes */
        color: #333333;
    }
    /* Structure des Titres */
    h1 {
        color: #2c3e50;
        font-size: 18pt;
        border-bottom: 2px solid #2c3e50;
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    h2 {
        color: #e67e22; /* Une couleur différente pour les sous-titres */
        font-size: 14pt;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    h3 {
        font-size: 12pt;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    /* Structure des Paragraphes */
    p {
        margin-bottom: 10px;
        text-align: justify;
    }
    /* Structure du Gras (Strong) */
    strong, b {
        color: #000;
        font-weight: bold;
    }
    /* Structure des Listes */
    ul {
        margin-bottom: 10px;
        padding-left: 20px;
    }
    li {
        margin-bottom: 5px; /* Espace entre les points */
        list-style-type: disc; /* Force les puces visibles */
    }
    /* Boîtes de code ou données */
    pre, code {
        background-color: #f4f4f4;
        font-family: Courier, monospace;
        padding: 2px;
        border-radius: 3px;
    }
</style>
"""

def convert_md_to_pdf(input_filename, output_filename):
    if not os.path.exists(input_filename):
        print(f"Erreur: Le fichier '{input_filename}' est introuvable.")
        return

    # 1. Lire le fichier Markdown
    with open(input_filename, "r", encoding="utf-8") as md_file:
        md_text = md_file.read()

    # 2. Convertir Markdown en HTML
    # L'extension 'extra' active les tableaux et les listes avancées
    # L'extension 'nl2br' transforme les retours à la ligne simples en sauts de ligne HTML
    html_body = markdown.markdown(md_text, extensions=['extra', 'nl2br'])

    # 3. Assembler le HTML complet avec le CSS
    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        {CSS_STYLE}
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """

    # 4. Générer le PDF
    try:
        with open(output_filename, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(
                src=html_content,
                dest=pdf_file,
                encoding='utf-8'
            )

        if pisa_status.err:
            print("Erreur lors de la génération du PDF.")
        else:
            print(f"Succès ! PDF structuré sauvegardé sous : '{output_filename}'")
    except Exception as e:
        print(f"Une exception est survenue : {e}")


# --- Usage Example ---
# if __name__ == "__main__":
#     # Create a test file if needed
#     if not os.path.exists("test.md"):
#         with open("test.md", "w", encoding="utf-8") as f:
#             f.write("# Pure Python PDF\n\nNo external tools required.\n\n## Features\n- Tables\n- Lists")
    
#     convert_md_to_pdf("trading_recommendation_20260122_022705.txt", "output_pure.pdf")