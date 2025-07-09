
#hi my name is abhijeet singh (@codebitbybit)
#  mrm8488/longformer-base-4096-finetuned-squadv2


from transformers import pipeline,AutoTokenizer,AutoModelForQuestionAnswering
from youtube_transcript_api import YouTubeTranscriptApi

from transformers.utils.logging import set_verbosity_error
set_verbosity_error()



context=""
video_url=input("enter video url:-")
print("please wait................")


        


id=video_url.split(".be/")

id=id[1].split("?")

video_id=id[0]

ytt_api = YouTubeTranscriptApi()

fetched_transcript = ytt_api.fetch(video_id )


for snippet in fetched_transcript:
        
        context+=(snippet.text)

# print(context)

modelid="mrm8488/longformer-base-4096-finetuned-squadv2"

model=pipeline(
    
      task="question-answering",

      model=modelid,

      device="cpu"    
    
      )

question=input("enter the question:-")
print("please wait........")
response=model(
      context=context,
      question=question
  )


print(f'the answer is:- \n{response.get("answer")}')


