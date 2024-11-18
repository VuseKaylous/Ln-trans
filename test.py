from crewai import Agent, Process, Task, Crew, LLM
# from langchain_community.llms import Ollama
# from langchain_ollama import ChatOllama

# os.environ["OPENAI_API_BASE"] = "http://localhost:11434"
# os.environ["OPENAI_MODEL_NAME"] = "llama3"
# os.environ["OPENAI_API_KEY"] = "NA"

# llm = ChatOllama(model="llama3")
# llm = LLM(model="ollama/llama3", base_url="http://localhost:11434")

llm_translator = LLM(model="ollama/llama3", base_url="http://localhost:11434")
llm_summarizer = LLM(model="ollama/test1", base_url="http://localhost:11434")
llm_writer = LLM(model="ollama/llama3", base_url="http://localhost:11434")

# Define your agents
translator = Agent(
    role="Translator",
    goal="Translate the document from English to Vietnamese",
    backstory="You are a language translation and cultural interpretation specialist, acting as the bridge between English and Vietnamese information streams. You ensure smooth communication across agents dealing with cross-cultural data by accurately translating both literal content and underlying contextual nuances",
    # allow_delegation=False,
    llm=llm_translator,
)

summarizer = Agent(
    role="Summarize",
    goal="Summarize the main ideas of the passage",
    backstory="You are an advanced summarization agent specializing in extracting essential information from vast narratives without missing critical details. You ensure that every summary you produce is concise yet complete, capturing the core ideas, key events, and subtle nuances necessary for informed decision-making.",
    # allow_delegation=False,
    llm=llm_summarizer,
)

writer = Agent(
    role="Writer",
    goal="Adapt the content of the translated text to the background context",
    backstory="You are a professional writing agent specializing in generating well-structured, compelling content across various formats. Whether crafting reports, articles, creative narratives, or technical documentation, you ensure every piece is clear, engaging, and tailored to its intended purpose.",
    # allow_delegation=False,
    llm=llm_writer,
)

# proofreader = Agent(
#     role="Proofreader",
#     goal="Check the sentence for any grammar, spelling, punctuation and formatting errors.",
#     backstory="You are an expert proofreader with extensive knowledge about grammar and linguistic. You are currently tasked to check if there are any errors in the given paragraph and fix it.",
#     allow_delegation=False,
#     llm=llm,
# )

# Define your task
translate_task = Task(
    description="Please provide a rough translation of the following paragraph: `{paragraph}`. Focus on conveying the primary meaning and key points without worrying about perfect grammar, idiomatic accuracy, or stylistic polish.",
    expected_output="A Vietnamese paragraph",
    agent=translator,
    # human_input=True,
)

summarize_task = Task(
    # description='Summarize the following paragraph without cutting any details, big or small: "{paragraph}". No need to write anything else.',
    description="Provide a detailed summary of the following paragraph: `{paragraph}`. Capture the main ideas, key points, and any essential details while avoiding redundancy. Ensure the summary is concise but thorough, accurately reflecting the original content.",
    expected_output="A paragraph that contains the summary of the provided paragraph.",
    agent=summarizer,
    # human_input=True,
)

writer_task = Task(
    # description='Rewrite each sentence of the following paragraph: {paragraph}, so that its content aligns better with the overall context provided: {summary}. Ensure that no sentence or information is removed. Maintain the original meaning while enhancing coherence, flow, and relevance to the given content.',
    description='Rewrite each sentence of the following paragraph: {paragraph}, so that its content aligns better with the overall context provided. Ensure that no sentence or information is removed. Maintain the original meaning while enhancing coherence, flow, and relevance to the given content.',
    expected_output="A Vietnamese paragraph.",
    agent=writer,
    # human_input=True,
)

# proofreader_task = Task(
#     description="Check the grammar of the paragraph and fix them if there is a mistake, also don't write anything else other than the translated paragraph",
#     expected_output="A paragraph.",
#     agent=proofreader,
# )

# -----------------------------------------------------------------------------------

# translate_crew = Crew(
#     agents=[translator],
#     tasks=[translate_task],
#     # process=Process.sequential
# )
#
# summary_crew = Crew(
#     agents=[summarizer],
#     tasks=[summarize_task],
#     # process=Process.sequential
# )
#
# writer_crew = Crew(
#     agents=[writer],
#     tasks=[writer_task],
#     # process=Process.sequential
# )

# -----------------------------------------------------------------------------------

# working_pipeline = Pipeline(
#     stages=[[translate_crew, summary_crew], writer_crew]
# )

# Define the manager agent
# manager = Agent(
#     role="Project Manager",
#     goal="Efficiently manage the crew and ensure high-quality translation",
#     backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed and the final translation is of highest quality",
#     allow_delegation=True,
#     llm=llm,
# )

# Instantiate your crew with a custom manager
crew = Crew(
    agents=[summarizer, writer],
    tasks=[summarize_task, writer_task],
    # manager_llm=llm,
    # process=Process.hierarchical,
    process=Process.sequential,
)

# Start the crew's work
inputs = {
    "jp_paragraph1": """
    俺は34歳住所不定無職。

　人生を後悔している真っ最中の小太りブサメンのナイスガイだ。

　つい三時間ほど前までは住所不定ではない、

　ただの引きこもりベテランニートだったのだが、

　気付いたら親が死んでおり、

　引きこもっていて親族会議に出席しなかった俺はいないものとして扱われ、

　兄弟たちの奸計にハマり、見事に家を追い出された。
    """,
    "jp_paragraph2": """
    床ドンと壁ドンをマスターし、

　家で傍若無人に振舞っていた俺に味方はいなかった。


　葬式当日、ブリッヂオ○ニー中にいきなり喪服姿の兄弟姉妹たちに部屋に乱入され、絶縁状を突きつけられた。

　無視すると、弟が木製バットで命よりも大切なパソコンを破壊しやがった。

　半狂乱で暴れてみたものの、兄は空手の有段者で、逆にぼっこぼこにされた。

　無様に泣きじゃくって事無きをえようとしたら、着の身着のまま家から叩き出された。
    """,
    "jp_paragraph3": """
    ズキズキと痛む脇腹（多分肋骨が折れてる）を抑えながら、とぼとぼと町を歩く。

　家を後にした時の、兄弟たちの罵詈雑言が未だ耳に残っている。

　聞くに堪えない暴言だ。

　心は完璧に折れていた。

　俺が一体なにをしたっていうんだ。

　親の葬式をブッチして無修正ロリ画像（兄の娘を風呂に入れた時にデジカメで撮りました）でオ○ってただけじゃないか……。
    """,
    "eng_paragraph": """
        “We should start back,” Gared urged as the woods began to grow dark around them. “The
        wildlings are dead.”
        “Do the dead frighten you?” Ser Waymar Royce asked with just the hint of a smile.
        Gared did not rise to the bait. He was an old man, past fifty, and he had seen the lordlings come
        and go. “Dead is dead,” he said. “We have no business with the dead.”
        “Are they dead?” Royce asked softly. “What proof have we?”
        “Will saw them,” Gared said. “If he says they are dead, that’s proof enough for me.”
        Will had known they would drag him into the quarrel sooner or later. He wished it had been later
        rather than sooner. “My mother told me that dead men sing no songs,” he put in.
        “My wet nurse said the same thing, Will,” Royce replied. “Never believe anything you hear at a
        woman’s tit. There are things to be learned even from the dead.” His voice echoed, too loud in
        the twilit forest.
        “We have a long ride before us,” Gared pointed out. “Eight days, maybe nine. And night is
        falling.”
        """,
    "eng_paragraph2": """
  "Not exactly," I say. As a general rule, and especially lately, I try to give Simon as little information as possible. We push through green metal doors to the back stairwell, a dividing line between the dinginess of the original Bayview High and its bright, airy new wing. Every year more wealthy families get priced out of San Diego and come fifteen miles east to Bayview, expecting that their tax dollars will buy them a nicer school experience than popcorn ceilings and scarred linoleum.""",
    "jp_sus_para1": """
    それでも子どもの頃から大好きだった従兄に会えるのは嬉しくて、彼の前に出るのだからとついおしゃれをしてしまった。といっても、金色の髪をきれいにまとめて自分の目と同じ落ち着いたグリーンのドレスに普段より高価なブローチを着けたくらいだが。
    """,
    "jp_sus_para2": """
    思わずそんな言葉が溢れる。
　俺だって、生まれた時からクズ人間だったわけじゃないのだ。
　そこそこ裕福な家庭の三男として生まれた。兄兄姉弟。５人兄弟の４番目。
　小学生の頃は、この歳にしては頭がいいと褒められて育った。
　勉強は得意じゃなかったが、ゲームがうまくて、運動もできるお調子者。
　クラスの中心だった。
　中学時代にはパソコン部に入り、雑誌を参考に、お小遣いを貯めて自作PCを作成。
　パソコンのパの字も知らなかった家族からは、一目も二目も置かれていた。
    """,
    "jp_sus_para3": """
    ワケあり女子の預かりが急激に増えたせいで、修道院の台所事情はより苦しいものとなった。
　厳しい気候のため、お布施や巡礼者が少ないのにも関わらず、大した支度金も無いまま、躾もモラルもなっていない欲しがり肉食獣を預かるこちらの身にもなって欲しい。
    """
}

input = {
    "paragraph": inputs["eng_paragraph2"],
    "tone" : "neutral"
}

# initial_translation = translate_crew.kickoff(inputs=input)
# print("Initial translation:")
# print()
# print(initial_translation)
# print()
# summary = summary_crew.kickoff(inputs=input)
# print("Summary:")
# print()
# print(summary)
# print()
# result = writer_crew.kickoff(inputs={"paragraph": initial_translation.raw, "summary": summary.raw})

result = crew.kickoff(inputs=input)
# result = working_pipeline.kickoff(input)

print(result.tasks_output[0])
print()
print("######################")
print()
print(result)
    

