from crewai import Agent, Task, Crew, LLM
import os
# os.environ["OPENAI_API_KEY"] = "NA"
os.environ["OLLAMA_HOST"] = "0.0.0.0"
    
llm = LLM(model="ollama/llama3", base_url="http://localhost:11434")

general_agent = Agent(role = "Translator",
                      goal = "Translate the given paragraph",
                      backstory = "You are a professional translation agent specializing in fast, accurate, and culturally sensitive translations across multiple languages. You ensure seamless communication between agents and external entities by maintaining the original meaning, tone, and intent of every message.",
                      allow_delegation = False,
                      verbose = True,
                      llm = llm)

task = Task(description="Translate the following paragraph from Japanese to Vietnamese. Ensure the translation accurately conveys the original meaning, including any cultural nuances or idiomatic expressions, while maintaining clarity and fluency in Vietnamese: {paragraph}",
             agent = general_agent,
             expected_output="A Vietnamese paragraph.")

crew = Crew(
    agents=[general_agent],
    tasks=[task],
)

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
    """
}

input = {
    "paragraph": inputs["jp_sus_para2"],
    "tone" : "neutral"
}

result = crew.kickoff(inputs=input)

print(result)
