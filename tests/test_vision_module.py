# Simple script to test VisionService
from services.vit_vision_service import VisionService

if __name__ == "__main__":

    # instanciate the service :
    vision = VisionService(
        references_dir="data/references"
    )

    # process pdf file :
    result = vision.process_document(
        pdf_path="data/references/cih-2.pdf"
    )

    print(result)

    


   