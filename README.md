# generative-ai-and-large-language-models
geeksvisor: Exploring the frontiers of generative AI and large language models. Our team delves into cutting-edge technology to unleash innovative solutions. Join us on this journey of limitless possibilities in artificial intelligence.
frst please gothorugh this article wrote  by Bill Gates "https://www.gatesnotes.com/The-Age-of-AI-Has-Begun"
===========================================================================================================================================================
our yesterday session summery (writen by muhammad mughess)
1. What are Large Language Models?
* LLMs are like super-smart machines that can generate content similar to what humans create.
* They're part of generative AI, a cool branch of machine learning.
* These models learn by studying tons of human-generated content, thanks to massive datasets and lots of computing power.
2. Foundation Models and Parameters:
* Foundation models, or base models, are the core of LLMs with billions of parameters (think of them as the model's memory).Parameters are the internal settings or variables within a machine learning model that the algorithm adjusts during training. They are like the knobs and dials that the model turns to fine-tune its understanding of the data.
* More parameters mean more memory and the ability to handle sophisticated tasks.
* Throughout our journey, we'll represent LLMs with purple circles. Cool, right?
3. Your Tool for the Course: Flan-T5
* In our labs, we'll use an open-source model called flan-T5 for language tasks.
* You can use these models as they are or fine-tune them for your specific needs—no need to start from scratch.
4. Interacting with Language Models:
* Unlike traditional coding, where you use formal syntax, LLMs understand natural language.
* You communicate with them through prompts—basically, human-written instructions.
* The prompt's space or memory is called the context window, and it's large enough for a few thousand words.
5. How Models Work:
* Example: Ask the model about Ganymede's location in the solar system.
* Your prompt goes in, the model predicts the next words, and voila! You get a completion.
* This process is called inference, and the output is a combination of your prompt and the generated text.
6.  Use Cases and tasks:
* LLMs can do more than chat. You can use them to write essays, summarize conversations, translate languages, or even generate code in Python.
* For instance, you can prompt a model to write code that calculates the mean of every column in a DataFrame.

===============================================================================================================================
Transformers Architecture

1. Machine Learning models machine-learning models are just big statistical calculators and they work with numbers, not words. So Tokenize the input words.
2 Input tokens are added to the encoder side of the network and passed through embedding layer and fed into multi head attention layers and in here it capturing contextual information from input sequences and passed deep representation to decor
3. Decoder generates new tokens based on input token triggers and encoder's contextual understanding.
4. Machine Learning models machine-learning models are just big statistical calculators and they work with numbers, not words. 
5. Remember that you'll be interacting with transformer models through natural language, creating prompts using written words, not code. You don't need to understand all of
the details of the underlying architecture to do this.
This is called prompt engineering.
 
Indepth Knowlege plesse study this article and video
"https://jalammar.github.io/illustrated-transformer/"
"https://www.youtube.com/watch?v=SMZQrJ_L1vo"

=======================================================================================================
What is Fine Tuning?

Fine-tuning in the context of machine learning, and specifically with large language models (LLMs), refers to the process of taking a pre-trained model and adjusting its parameters to make it more suited for a specific task or domain. Let's break it down:
1. Pre-trained Model:
* Before fine-tuning, a model is pre-trained on a large and diverse dataset. This initial training helps the model learn general patterns and features from a broad range of data.
2. Fine-tuning Process:
* After pre-training, fine-tuning involves taking the pre-trained model and training it further on a smaller, more specific dataset related to the task at hand.
* The process involves exposing the model to task-specific examples and adjusting its parameters to make it more specialized in handling the nuances of the particular domain or task.
3. Why Fine-tune?
* Fine-tuning is beneficial when you have a specific task that may differ from the original pre-training data.
* It allows you to leverage the knowledge gained during pre-training while adapting the model to perform well on a narrower, task-specific dataset.
4. Examples of Fine-tuning:
* In the context of large language models:
    * You might fine-tune a language model on a dataset related to medical texts to make it better at generating medical content.
    * Fine-tuning can also involve adjusting the model for sentiment analysis, code generation, summarization, or any specific language-related task.
5. Fine-tuning Smaller Models:
* Even smaller models with fewer parameters can be fine-tuned for specific tasks, and this process is often quicker than training a model from scratch.


sesssion date(12/12/23)
In our labs we are using Base model T-5 encoder-decoder-model to train out model

-prompting and prompt engineering
.Prompting in LLM Training:
In LLM training, prompting involves exposing the model to diverse text data during the learning process. These prompts, comprising sentences and paragraphs, help the model understand language patterns and context.
.Prompt Engineering in LLM Training:
Prompt engineering in LLM training focuses on optimizing the training data. It includes tasks such as cleaning data, introducing variations for robustness, designing task-specific prompts, balancing data representation, and leveraging transfer learning for fine-tuning. Effective prompt engineering enhances the model's performance and generalization.
![image](https://github.com/abdulrehman035551/generative-ai-and-large-language-models/assets/96192529/8f1d4536-c121-4402-9944-57dbe913d161)
please check this image for better understaing 
1-Zero-shot Inference:
Zero-shot prompting refers to a methodology wherein a substantial language model (LLM) is tasked with executing a task for which it has not received explicit training.
![image](https://github.com/abdulrehman035551/generative-ai-and-large-language-models/assets/96192529/77a3fd3e-e08a-4f6b-b686-1beac03f5866)
please check this image for better understaing 
2-One-shot Inference:
Remember, this is needed in a small model. If we pass an example to our model along with our question, it aids in better understanding of the prompt.
![image](https://github.com/abdulrehman035551/generative-ai-and-large-language-models/assets/96192529/38c02967-54ed-41a7-a860-1ffc33d8ae6c)
please check this image for better understaing 
3-Few-shot Inference:
Remember, this is needed in an even-smaller model.  we pass many examples to our model along with our question, it enhances understanding of the prompt.
![image](https://github.com/abdulrehman035551/generative-ai-and-large-language-models/assets/96192529/14a68a6b-e617-4ee8-9347-3eda58800173)
please check this image for better understaing 











