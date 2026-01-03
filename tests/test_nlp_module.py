from services.nlp_service import NLPModule


if __name__ == "__main__":

    module = NLPModule(
        keywords_path="data/keywords.txt" , 
        keyword_threshold=3
    )


    result = module.process_document(
        pdf_path="data/references/cih-2.pdf"
    )

    print(result)
    


   