from education2skill import Education2Skill
import pandas as pd
import numpy as np

def predictions_from_text(education2skill):    
    input_text = "General description:Aim (s) of the study program:To train highly qualified universal IT specialists who are able to design, develop and maintain software: to create formal information models in the field of application or to use existing ones to achieve predefined goals; to implement the project independently or in a multicultural group with modern program systems development tools and technologies; evaluate the developed software in terms of efficiency, correctness, security and scalability; qualified installation, operation and updating of hardware and software for computers and their systems.Study results:Teaching and learning activities:Methods of assessment of study results:Structure:Study subjects (modules), practice:The study program consists of subjects of the study field - 70.8%, electives - 16.7%, internship in enterprises - 6.25%, general university study subjects - 6.25%.Specializations:Student choices:ā€¢Distinctive features of the study program:The study program allows you to choose subjects from a wide list of options or create an individual study plan. Based on the classical understanding of computer science, the program includes theoretical computer science disciplines (algorithms, programming languages, operating systems, and computational models) and application modules (design, implementation, and application of software systems in business and industry). The content of the program reflects the relevant recommendations of ACM, IEEE CS organizations (Computer Science Curriculum).Professional activities and further study opportunities:Professional opportunities:Graduates are prepared to work as analysts, designers, programmers and testers in large-scale projects, manufacturing companies and organizations providing complex, science-intensive software products and software solutions.Further study opportunities:Graduates of the program can continue their studies in informatics, information systems, software systems and other master's degree programs of mathematics and computer science in Lithuanian or foreign higher education institutions."
    
    skills_prediction = education2skill.skills_from_single_description(input_text)
    
    for level, _, probability, (lpk, skill) in skills_prediction:
        print (level * " ", lpk, skill, probability)        
    print ()
    
def predictions_from_single_embedding(education2skill):
    input_text = "General description:Aim (s) of the study program:To train highly qualified universal IT specialists who are able to design, develop and maintain software: to create formal information models in the field of application or to use existing ones to achieve predefined goals; to implement the project independently or in a multicultural group with modern program systems development tools and technologies; evaluate the developed software in terms of efficiency, correctness, security and scalability; qualified installation, operation and updating of hardware and software for computers and their systems.Study results:Teaching and learning activities:Methods of assessment of study results:Structure:Study subjects (modules), practice:The study program consists of subjects of the study field - 70.8%, electives - 16.7%, internship in enterprises - 6.25%, general university study subjects - 6.25%.Specializations:Student choices:ā€¢Distinctive features of the study program:The study program allows you to choose subjects from a wide list of options or create an individual study plan. Based on the classical understanding of computer science, the program includes theoretical computer science disciplines (algorithms, programming languages, operating systems, and computational models) and application modules (design, implementation, and application of software systems in business and industry). The content of the program reflects the relevant recommendations of ACM, IEEE CS organizations (Computer Science Curriculum).Professional activities and further study opportunities:Professional opportunities:Graduates are prepared to work as analysts, designers, programmers and testers in large-scale projects, manufacturing companies and organizations providing complex, science-intensive software products and software solutions.Further study opportunities:Graduates of the program can continue their studies in informatics, information systems, software systems and other master's degree programs of mathematics and computer science in Lithuanian or foreign higher education institutions."
    
    embedding = education2skill.bert.make_single_embedding(input_text)   
    
    skills_prediction = education2skill.skills_from_single_embedding(embedding)
    
    for level, _, probability, (lpk, skill) in skills_prediction:
        print (level * " ", lpk, skill, probability)        
    print ()
    
def predictions_from_multiple_embeddings(education2skill):
    def get_embeddings():
        data = pd.read_csv("./input/test_and_predict.csv")
        for i in range(len(data.iloc[:,0])):
            yield np.asarray(data.iloc[i,4:]).astype('float32').reshape(1,-1)
    
    multiple_embedding_predictions = education2skill.skills_from_embeddings(get_embeddings())
        
    for embedding, skills_prediction in multiple_embedding_predictions:
        for level, _, probability, (code, title) in skills_prediction:
            print (level * " ", code, title, probability)        
        print ()
    

if __name__ == '__main__':
    model_filename = "./models/finaldnau1000withoutweightsseq.h5"
    lpk_index_map_filename = './models/lpkindexmap.json'
    
    education2skill = Education2Skill(model_filename, lpk_index_map_filename)
    
    predictions_from_text(education2skill)
    
    predictions_from_single_embedding(education2skill)
    
    #predictions_from_multiple_embeddings(education2skill)
    
    
