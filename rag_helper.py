
INSTRUCTIONS = '''
            Your task is to answer the question based on the context provided.
            Use the following context to answer the question. 
            If you don't know the answer, say you don't know. 
            Do not try to make up an answer.

'''


USER_PROMPT_TEMPLATE = '''   
            
            Question: {question}
            Context: {context}
            


     '''

class RAGBase:
    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        course='llm-zoomcamp',
        model='gpt-5.4-mini',
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.course = course
        self.model = model

    def search(self, question, course='llm-zoomcamp'):
        boost_dict = {'question': 2.0, 'section': 0.5}
        filter_dict = {'course': self.course}

        return self.index.search(
            question,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
            num_results=5,
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc.get('section', ''))
            lines.append('Q:' + doc.get('question', ''))
            lines.append('A:' + doc.get('answer', ''))
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, question, search_results):
        context = self.build_context(search_results)
        prompt = self.prompt_template.format(
            question=question,
            context=context,
        )
        return prompt.strip()

    def llm(self, instructions, user_prompt, model=None):
        if model is None:
            model = self.model

        message_history = [
            {'role': 'developer', 'content': instructions},
            {'role': 'user', 'content':user_prompt},
        ]

        response = self.llm_client.responses.create(
            model=model,
            input=message_history,
        )

        # depending on client, output_text may be attribute or key
        return response.output_text 
    

    def rag(self,query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(self.instructions, prompt)
        return answer 



