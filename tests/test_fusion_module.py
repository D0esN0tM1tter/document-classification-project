from services.fusion_service import MultimodalDocumentClassifier


if __name__ == "__main__":

    module = MultimodalDocumentClassifier()

    result = module.process_document(
        pdf_path="data/references/cih-2.pdf"
    )

    print(result)

   