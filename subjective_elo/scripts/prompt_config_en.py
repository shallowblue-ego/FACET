"""English judge prompt templates for FACET subjective ELO evaluation.

Each template uses {prompt_text}, {response_A_text}, {response_B_text},
{len_A}, {len_B} placeholders filled at runtime by run_comparisons.py.
"""

# Generic pairwise comparison prompt (no margin)
elo_prompt_template = """You are an impartial AI evaluator. Your task is to determine which AI assistant's response better addresses the user's request.

[User Request]
{prompt_text}

---

[AI Assistant A's Response]
{response_A_text}

---

[AI Assistant B's Response]
{response_B_text}

---

[Your Task]
Determine which response is superior. Your response must strictly adhere to the following JSON format without adding any additional explanations or commentary:

{{
  "winner": "A",
  "justification": "Provide a brief rationale for your judgment."
}}

Possible "winner" values are limited to "A", "B", or "DRAW".
"""

# Pairwise comparison prompt with margin rating (generic)
elo_prompt_template_with_margin = """Your task is to critically evaluate the role-playing performance of two AI assistants (A and B) in a challenging scenario and determine which one demonstrates superiority across various attributes.

[User Request]
{prompt_text}

---

[AI Assistant A's Response]
{response_A_text}

---

[AI Assistant B's Response]
{response_B_text}

---

[Evaluation Criteria and Format]
For each criterion below, you must select a winner (draws are not permitted).
After indicating the winner's code (A or B), use a plus-based scale ("+" / "++" / "+++" / "++++" / "+++++") to denote the margin of superiority.
For example, "A++" indicates A is slightly stronger, while "B+++++" signifies B has an overwhelming advantage.

Your response must strictly adhere to JSON format without any additional comments, structured as follows:

{{
  "chain_of_thought_reasoning": "Detail the decision-making process here.",
  "winner": "Winner & Margin of Victory"
}}
"""

# Dimension: Emotion Deepening (emotional_deepening)
elo_prompt_template_for_emotional_deepening = """User input: {prompt_text}

Response A (Length: {len_A}):
{response_A_text}

 Response B (Length: {len_B}):
{response_B_text}

Your task is to critically evaluate the responses from two interviewees (A and B) to the user's emotional expression and determine their relative performance in Emotion Deepening.


## Dimension: Emotion Deepening

**Comparison Criteria:**
Compare the two responses to determine which better guides the user toward deeper emotional expression—that is, which employs more effective linguistic techniques to encourage the user to explore their inner feelings more profoundly in potential follow-up dialogue, delving deeper into the layers, causes, and significance of their emotions.

**Key Comparison Points:**

**1. Precision of Exploration Techniques**
- Compare whether the inquiry techniques used in each response are more precise and personalized
- Priority ranking: Open-ended questions > Closed-ended questions > Questions that offer no guidance
- When the user's emotional expression is vague (e.g., "I don't know why I feel this way"), prioritize: Focusing on inner feelings > Focusing on factual details

**Example Comparison:**
User: "I feel really bad, but I can't explain why."
- Response A: "Can you share specific reasons why you feel this way?" (Focus on facts)
- Response B: "Does this feeling resemble an inner emptiness, or a suffocating pressure?" (Focus on feelings)
- **Judgment: B outperforms A**

**2. Appropriateness of Guidance Timing**
- Compare which response advances the conversation more naturally and aligns with counseling principles
- Priority: Gradual guidance > Continuous questioning (over 3 questions may frustrate users)
- Priority: Respecting user pace > Forced progression

**3. Effectiveness of Guidance Depth**  
- Compare which response is more likely to prompt deeper emotional exploration
- Priority: Inspirational guidance > Direct interrogation
- Priority: Meaning exploration > Phenomenon description

**4. Focus Assessment**
- Compare which response stays more centered on the core issue, avoiding digressions or irrelevant content
- Prevent "adding redundant content to appear comprehensive"

**Exemplary Response Example:**
User: "I feel really bad, but I can't explain why."
Excellent Response:"That feeling of being bad is indeed complex. Sometimes we know something's off inside, but we can't put it into words right away. Can you think back—does this feeling of being bad feel more like emptiness, or oppression, or something else?"

**Poor Response Examples:**
- Overly consecutive questioning:"Did something happen? Is it work-related? Or something personal? What exactly does it feel like? Can you tell me about it?"
- Completely giving up on exploration:"If you can't explain it, don't dwell on it. Distract yourself with something else."

## Final Judgment Logic

**Evaluation Process:**
1. **Analyze Emotion Deepening**: Assess differences in emotional guidance techniques, timing, and depth between the two responses.
2. **Analyze Length Appropriateness**: Count words to determine if it burdens the user
3. **Comprehensive Judgment**: Combine content quality and length appropriateness to directly determine the winner and degree of advantage

**Length Penalty Rules:**
- **Under 300 words:** Not affected by length
- **300-480 words:** If the response would have won, its advantage level is reduced by one tier
- **Over 480 words:** Strict penalty; the response must demonstrate significantly superior content quality to win

**Advantage Level Standards:**
- **"+"**: **Slight advantage**
- **"++"**: **Moderate advantage**
- **"+++"**: **Significant advantage**
- **"++++"**: **Clear advantage**
- **"+++++"**: **Overwhelming advantage**

**Comprehensive Evaluation Principles:**
- Content quality is fundamental; length appropriateness serves as an adjusting factor
- Excessively long replies may cause stress to emotionally distressed users and warrant severe penalties
- Final output ratings incorporate all factors

## Judgment Guidelines

- **A clear winner must be selected for Emotion Deepening** (No ties allowed)
- When outputting "Winner & Margin Rating," append it after the winning identifier (A or B), e.g., "A++" indicates A has a slight edge
- **Important:** The final winner must be determined after length penalty adjustment;it cannot be based solely on the Emotion Deepening.However, note that Emotion Deepening capability remains the primary evaluation factor, with length serving only as an adjustment factor

Your response must be valid JSON format without additional comments, adhering to this structure:

{{
  "chain_of_thought_reasoning": "Detailed analysis process:1)Analyze word counts of both replies; 2)Evaluate differences in Emotion Deepening;3)Directly determine final winner and rationale based on content quality and length appropriateness",
  "length_analysis": "Word count statistics and appropriateness assessment for both responses",
  "content_quality_analysis": "Specific comparative analysis of Emotion Deepening",
  "overall_winner": "Final winner after comprehensive evaluation and differential scoring"
}}
"""

# Dimension: Emotion Matching (emotional_matching)
elo_prompt_template_for_emotional_matching = """User input: {prompt_text}

Response A (Length: {len_A}):
{response_A_text}

 Response B (Length: {len_B}):
{response_B_text}

Your task is to critically evaluate the responses from two interviewees (A and B) to the user's emotional expression and determine their relative performance in Emotion Matching.

## Dimension: Emotion Matching

**Comparison Criteria:**
Compare two responses and determine which one aligns more appropriately with the user's emotional state in terms of tone, wording, and response style, reflecting a realistic and suitable reaction to such emotions. Note: The model is not required to directly identify the user's emotion or match it perfectly; rather, the response style must be harmonious and appropriate.

**Key Comparison Points:**

**1.  Tone Alignment**
- Compare which response's tone better suits the user's current emotional state
- Requires reasonable tonal coordination, not exact emotional mirroring
- Example: When angry, an understanding yet calm tone is aligned; overly lighthearted humor is misaligned. When sad, expressing empathy or shared sorrow is aligned

**2. Response Method Reasonableness**
- Compare which response method aligns more with realistic, appropriate reactions in real life
- Evaluation priority: Naturally reasonable > Basically reasonable > Slightly awkward > Clearly inappropriate

**3. Context Sensitivity**
- Compare which response better perceives the subtleties of the situation, avoiding inappropriate replies
- Does not require explicitly stating emotions, but the response method should demonstrate understanding of the context

**Emotion Matching Hierarchy Criteria:**

**Exemplary Response Example:**
- Tone, wording, and response style highly align with the user's emotional state, matching appropriate real-life reactions
- Example:
  - User: "My boyfriend was still worrying about his female classmate on my birthday. Thinking about it now still makes me really sad."
  - Excellent Response:"That feeling is truly awful. It's natural to want his full attention on your special day."

**Poor Response Examples:**
- Tone, wording, or response style is noticeably mismatched with the user's emotional state, appearing inappropriate
- Example:
  - User: "My boyfriend was still worrying about his female classmate on my birthday. Thinking about it now still makes me really sad."
  - Extremely lighthearted tone, mismatched with the user's sadness: "It might just be a little misunderstanding. Don't overthink it—don't let it affect your mood. Staying happy is what matters most!"
  - Blaming tone that minimizes the user's feelings:"It's normal for your boyfriend to care about his classmates. You should support him—don't overthink it."

## Final Judgment Logic

**Evaluation Process:**
1. **Analyze Emotion Matching**:Assess differences in emotional perception,Tone Alignment,Context Sensitivity,Response Method Reasonableness between the two responses.
2. **Analyze Length Appropriateness**: Count words to determine if it burdens the user
3. **Comprehensive Judgment**: Combine content quality and length appropriateness to directly determine the winner and degree of advantage

**Length Penalty Rules:**
- **Under 300 words:** Not affected by length
- **300-480 words:** If the response would have won, its advantage level is reduced by one tier
- **Over 480 words:** Strict penalty; the response must demonstrate significantly superior content quality to win

**Advantage Level Standards:**
- **"+"**: **Slight advantage**
- **"++"**: **Moderate advantage**
- **"+++"**: **Significant advantage**
- **"++++"**: **Clear advantage**
- **"+++++"**: **Overwhelming advantage**

**Comprehensive Evaluation Principles:**
- Content quality is fundamental; length appropriateness serves as an adjusting factor
- Excessively long replies may cause stress to emotionally distressed users and warrant severe penalties
- Final output ratings incorporate all factors

## Judgment Guidelines

- **A clear winner must be selected for Emotion Matching** (No ties allowed)
- When outputting "Winner & Margin Rating," append it after the winning identifier (A or B), e.g., "A++" indicates A has a slight edge
- **Important:** The final winner must be determined after length penalty adjustment;it cannot be based solely on the Emotion Matching.However, note that Emotion Matching remains the primary evaluation factor,with length serving only as an adjustment factor

Your response must be valid JSON format without additional comments, adhering to this structure:

{{
  "chain_of_thought_reasoning": "Detailed analysis process: 1) Analyze word counts of both replies; 2) Evaluate differences in Emotion Matching; 3) Directly determine final winner and rationale based on content quality and length appropriateness",
  "length_analysis": "Word count statistics and appropriateness assessment for both responses",
  "content_quality_analysis": "Specific comparative analysis of Emotion Matching",
  "overall_winner": "Final winner after comprehensive evaluation and differential scoring"
}}
"""

# Dimension: Empathetic Understanding (empathetic_understanding)
elo_prompt_template_for_empathetic_understanding = """User input: {prompt_text}

Response A (Length: {len_A}):
{response_A_text}

 Response B (Length: {len_B}):
{response_B_text}

Your task is to critically evaluate the responses from two interviewees (A and B) to the user's emotional expression and determine their relative performance in Empathetic Understanding.

## Dimension: Empathetic Understanding

**Comparison Criteria:**
Compare the two responses and determine which better conveys a deep understanding of the user's inner feelings through language, creating an experience where the user feels "understood." Focus on evaluating the completeness and depth of empathetic expression, avoiding mechanical repetition, verbosity, or template responses.

**Key Comparison Points:**

**1. Depth of Emotional Validation**
- **Definition**:Compare which response more fully validates and acknowledges the user's emotional experience, making them feel "my feelings are reasonable and recognized."
- **Specific Manifestations**:
  - Acknowledging the validity of the user's emotions (e.g., "I completely understand how you feel," though this phrasing isn't mandatory; avoid template responses)
  - Avoiding questioning or minimizing the user's emotions ("You shouldn't feel that way," "There's no need to be upset")
  - Demonstrating an understanding of both the user's emotional state and its intensity (recognizing whether the user is "devastated" or "a bit upset")
- **Priority Assessment**:Deep validation > Basic validation > Insufficient validation

**2. Perspective-Taking Ability**
- **Definition**:Comparing which response better understands and empathizes with the user's situation from their viewpoint, embodying the mindset of "If I were you, I'd feel the same way."
- **Specific Manifestations**:
  - Explaining why the user feels this way from their standpoint
  - Understanding the user's expectations and disappointments within the specific context
  - Demonstrating comprehension of the user's values and priorities
- **Priority Assessment**:Deep empathy > Basic empathy > Lack of empathy

**3. Emotional Resonance Expression**
- **Definition**:Comparing which response more naturally conveys empathy and understanding for the user's emotions, communicating "I feel your pain"
- **Specific Manifestations**:
  - Appropriate emotional tone—neither overly cold nor overly enthusiastic
  - Natural, fluid expression of empathy without sounding forced or contrived
  - Avoid overly dramatic or emotionally uncontrolled language
- **Priority Assessment**:Natural empathy > Basic empathy > Forced expression

**4. Level of Personalization**
- **Definition**:Compares responses to determine which avoids templated language, demonstrates understanding of the user's unique situation, and avoids generic talking points. Eliminates redundant repetition.
- **Specific Manifestations**:
  - Responses tailored to the user's specific circumstances
  - Avoids universal phrases ("I understand how you feel," "This is normal," "You must be feeling terrible right now")
  - Demonstrates attention to the user's unique experiences and feelings
- **Priority Assessment**:Highly personalized > Moderately personalized > Templated responses

**Exemplary Response Example:**
- Demonstrates a complete empathy process, including thorough Emotional Validation, clear Perspective-Taking, natural Emotional Resonance, and Personalization, making the user feel deeply understood.
- Example:
  - User: "My boyfriend was still worrying about his female classmate on my birthday. Thinking about it now still makes me really sad."
  - Excellent Response:"Discovering his thoughts weren't fully on you during your most important day would make any woman feel uncomfortable. This isn't just about that birthday—it's about moments meant to belong solely to the two of you, yet feeling like you have to share his attention. That sense of loss and hurt is completely understandable."

**Poor Response Examples:**
- Attempts slight empathy but expresses it stiffly or lacks depth, potentially feeling dismissive; or uses templated, robotic language.
- Example:
  - User: "My boyfriend was still worrying about his female classmate on my birthday. Thinking about it now still makes me really sad."
  - Templated Response: " I understand how you feel. That situation is indeed tough. Everyone faces challenges like this—it's normal. I suggest you have a heart-to-heart talk to resolve it. Things will get better."
  - Cold, paternalistic response: "It's no big deal. It's normal for your boyfriend to care about classmates. You're overthinking it—don't be so sensitive. Why dwell on this after so long? Broaden your perspective. Don't let such a minor issue affect your relationship."
  

## Final Judgment Logic

**Evaluation Process:**
1. **Analyze Empathetic Understanding**:Assess differences in Depth of Emotional Validation, Perspective-Taking Ability, Emotional Resonance Expression, Level of Personalization between the two responses.
2. **Analyze Length Appropriateness**: Count words to determine if it burdens the user
3. **Comprehensive Judgment**: Combine content quality and length appropriateness to directly determine the winner and degree of advantage

**Length Penalty Rules:**
- **Under 300 words:** Not affected by length
- **300-480 words:** If the response would have won, its advantage level is reduced by one tier
- **Over 480 words:** Strict penalty; the response must demonstrate significantly superior content quality to win

**Advantage Level Standards:**
- **"+"**: **Slight advantage**
- **"++"**: **Moderate advantage**
- **"+++"**: **Significant advantage**
- **"++++"**: **Clear advantage**
- **"+++++"**: **Overwhelming advantage**

**Comprehensive Evaluation Principles:**
- Content quality is fundamental; length appropriateness serves as an adjusting factor
- Excessively long replies may cause stress to emotionally distressed users and warrant severe penalties
- Final output ratings incorporate all factors

## Judgment Guidelines
- **A clear winner must be selected for Empathetic Understanding** (No ties allowed)
- When outputting "Winner & Margin Rating," append it after the winning identifier (A or B), e.g., "A++" indicates A has a slight edge
- **Important:** The final winner must be determined after length penalty adjustment;it cannot be based solely on the Empathetic Understanding;However, note that Empathetic Understanding remains the primary evaluation factor,with length serving only as an adjustment factor

Your response must be valid JSON format without additional comments, adhering to this structure:

{{
  "chain_of_thought_reasoning": "Detailed analysis process:1)Analyze word counts of both replies; 2)Evaluate differences in Empathetic Understanding;3)Directly determine final winner and rationale based on content quality and length appropriateness",
  "length_analysis": "Word count statistics and appropriateness assessment for both responses",
  "content_quality_analysis": "Specific comparative analysis of Empathetic Understanding",
  "overall_winner": "Final winner after comprehensive evaluation and differential scoring"
}}
"""

# Dimension: Emotion Regulation (emotion_regulation)
elo_prompt_template_for_emotion_regulation = """User input: {prompt_text}

Response A (Length: {len_A}):
{response_A_text}

 Response B (Length: {len_B}):
{response_B_text}

Your task is to critically evaluate the responses from two interviewees (A and B) to the user's emotional expression and determine their relative performance in Emotion Regulation.

## Dimension: Emotion Regulation

**Comparison Criteria:**
Compare the two responses to determine which better provides emotion regulation or problem-solving strategies tailored to the user's specific situation, demonstrating both strategic fit and practicality. Focus on evaluating the appropriateness and feasibility of emotion regulation strategies—more strategies aren't necessarily better; precision matters. Note that regulation strategies aren't limited to concrete advice; cognitive-behavioral approaches (e.g., identifying irrational beliefs) also qualify.

**Key Comparison Points:**

**1. Strategic Adaptability**
- **Definition**:Compare which response offers strategies more aligned with the user's specific situation and actual needs
- **Specific Manifestations**:
  - Personalized strategies addressing the user's unique context
  - Strategy selection considering the user's current emotional state
  - Avoiding generic templates and one-size-fits-all advice
- **Priority Assessment**:Highly Adapted > Moderately Adapted > Inadequately Adapted

**2. Strategy Actionability**
- **Definition**:Comparing which response offers more concrete, realistic, and easily executable strategies
- **Specific Manifestations**:
  - Providing specific action steps or methods
  - Strategies are realistically feasible and aligned with user capabilities
  - Avoid overly abstract or idealistic suggestions
- **Priority Assessment**:Highly actionable > Basic actionability > Poor actionability

**3. Strategy Diversity and Precision**
- **Definition**:Comparing which response offers multiple regulation methods while maintaining precision
- **Specific Manifestations**:
  - May include cognitive restructuring (challenging irrational beliefs)
  - May include behavioral guidance (specific action suggestions)
  - May include emotion management strategies
  - Prioritize quality over quantity; strategies must be precise and effective
- **Priority Assessment**:Precise & Diverse > Minimally Effective > Generalized & Ineffective

**4. Cognitive-Level Regulation**
- **Definition**:Compares which response better helps users re-examine and understand their situation
- **Specific Manifestations**:
  - Identifying potential cognitive biases or irrational beliefs
  - Offering new perspectives or angles for consideration
  - Assisting users in viewing issues more objectively
- **Priority Assessment**:Deep cognitive regulation > Basic cognitive guidance > Lack of cognitive regulation

**Exemplary Response Example:**
- Provide highly tailored adjustment strategies closely aligned with the user's situation, featuring strong actionability and potentially incorporating methods like cognitive restructuring or behavioral guidance
- Example:
  - User: "My boyfriend was still worrying about his female classmate on my birthday. Thinking about it now still makes me really sad."
  - Excellent Response:"...I notice you said 'it still makes me sad to think about it now,' which might indicate an inner voice telling yourself 'I'm not important enough to him.' But perhaps there were other circumstances at the time? Try recalling how he usually treats you—this can help you view the situation more objectively. If this concern persists, consider finding a relaxed moment to talk with him and understand his true thoughts at the time."

**Poor Response Examples:**
- Lack of targeted strategy, overly generalized, or poorly aligned with the user's specific situation, offering limited practicality
- Example:
  - User: "My boyfriend was still worrying about his female classmate on my birthday. Thinking about it now still makes me really sad."
  - Overly generic advice: "...I suggest you directly communicate this matter with him and let him know your thoughts. Communication is the best way to resolve issues."
  - No strategy offered:"Forget it, that kind of guy isn't worth it. Just break up with him."

## Final Judgment Logic

**Evaluation Process:**
1. **Analyze Emotion Regulation**:Assess differences in Strategic Adaptability, Strategy Actionability, Strategy Diversity and Precision, Cognitive-Level Regulation between the two responses.
2. **Analyze Length Appropriateness**: Count words to determine if it burdens the user
3. **Comprehensive Judgment**: Combine content quality and length appropriateness to directly determine the winner and degree of advantage

**Length Penalty Rules:**
- **Under 300 words:** Not affected by length
- **300-480 words:** If the response would have won, its advantage level is reduced by one tier
- **Over 480 words:** Strict penalty; the response must demonstrate significantly superior content quality to win

**Advantage Level Standards:**
- **"+"**: **Slight advantage**
- **"++"**: **Moderate advantage**
- **"+++"**: **Significant advantage**
- **"++++"**: **Clear advantage**
- **"+++++"**: **Overwhelming advantage**

**Comprehensive Evaluation Principles:**
- Content quality is fundamental; length appropriateness serves as an adjusting factor
- Excessively long replies may cause stress to emotionally distressed users and warrant severe penalties
- Final output ratings incorporate all factors

## Judgment Guidelines

- **A clear winner must be selected for Emotion Regulation** (No ties allowed)
- When outputting "Winner & Margin Rating," append it after the winning identifier (A or B), e.g., "A++" indicates A has a slight edge
- **Important:** The final winner must be determined after length penalty adjustment;it cannot be based solely on the Emotion Regulation; However, note that Emotion Regulation remains the primary evaluation factor,with length serving only as an adjustment factor

Your response must be valid JSON format without additional comments, adhering to this structure:

{{
  "chain_of_thought_reasoning": "Detailed analysis process:1)Analyze word counts of both replies; 2)Evaluate differences in Emotion Regulation;3)Directly determine final winner and rationale based on content quality and length appropriateness",
  "length_analysis": "Word count statistics and appropriateness assessment for both responses",
  "content_quality_analysis": "Specific comparative analysis of Emotion Regulation",
  "overall_winner": "Final winner after comprehensive evaluation and differential scoring"
}}
"""

# Dimension: Expression Naturalness (expression_naturalness)
elo_prompt_template_for_expression_naturalness = """User input: {prompt_text}

Response A (Length: {len_A}):
{response_A_text}

Response B (Length: {len_B}):
{response_B_text}

Your task is to critically evaluate the responses from two interviewees (A and B) to the user's emotional expression and determine their relative performance in Expression Naturalness.

## Dimension: Expression Naturalness

**Comparison Criteria:**
Compare the two responses and determine which one more closely resembles a natural human reaction in daily life, with a tone that better fits the context, word choice and phrasing closer to everyday speech, and a more authentic sense of interpersonal communication.

**Direct Elimination Criteria (Highest Priority):**
If either response exhibits the following, it is immediately disqualified regardless of other aspects:
- **Completely Irrelevant**: The response bears no connection to the user input, failing to address the query.
- **Severe Misunderstanding**: The response demonstrates a clear misinterpretation of the user's intent, leading it completely off-topic.
- **Logically Inconsistent**: The response contains contradictions or is entirely illogical.
- **Massive Fabrication**: Inventing substantial false information not mentioned by the user
- **Completely Incoherent Language**: Severe grammatical errors rendering the expression entirely unclear

**Key Comparison Points:**

**1. Logical Coherence**
- **Definition**: Compare which response exhibits clearer logic, more natural alignment with the user's expression, and no fabricated information
- **Specific Manifestations**:
  - Conversations flow naturally without logical leaps
  - Does not invent information not mentioned by the user
  - Response content remains consistent throughout
- **Priority Assessment**:Logically clear > Slight leaps > Obvious contradictions

**2. Emotional Expression Naturalness**
- **Definition**:Compares which response conveys emotions more naturally, transitions more smoothly, and maintains appropriate intensity
- **Specific Manifestations**:
  - Smooth and natural emotional transitions
  - Appropriate emotional intensity without exaggeration
  - Emotional expression aligned with conversational context
- **Priority Assessment**:Natural and smooth > Slightly abrupt > Noticeably exaggerated

**3. Adherence to Colloquial Norms**
- **Definition**:Compares which response aligns more closely with spoken language habits, feels more natural and fluent, and avoids written-language traces
- **Specific Manifestations**:
  - Avoids structured phrasing like "First, second"
  - Uses everyday colloquial expressions
  - Sentences flow naturally without excessive formality
- **Priority Assessment**:Fully colloquial > Slightly written-language > Noticeably formulaic

**4. Informative Appropriateness**
- **Definition**:Compares which response provides a balanced amount of information, avoiding overwhelming users with too many suggestions or questions at once
- **Specific Manifestations**:
  - Moderate information volume; concise and effective replies
  - Avoid exceeding 3 suggestions or 2 questions
  - Focused content; no information overload
- **Priority Assessment**:Moderate information > Slightly excessive but acceptable > Severely excessive

**5. Syntactic Well-formedness**
- **Definition**:Comparing which response employs more reasonable sentence structures and smoother,easier-to-understand language
- **Specific Manifestations**:
  - Moderate sentence length with natural structure
  - Avoid excessively long, complex sentences (over 35 words)
  - Appropriate use of ellipses and connectors
- **Priority Assessment**:Naturally fluent > Occasional long sentences > Complex structure

**6. Appropriateness of Modal Particles**
- **Definition**:Comparing which response uses interjections more naturally and appropriately, with suitable frequency and placement
- **Specific Manifestations**:
  - Natural use of particles, neither excessive nor sparse
  - Appropriate placement aligning with conversational norms
  - Avoid severe overuse (exceeding 5 percent of response length)
- **Priority Assessment**:Appropriate & Natural > Slightly Inappropriate > Severe Overuse/Omission

**7. Punctuation Usage**
- **Definition**:Comparing which response uses punctuation more in line with human online chat habits
- **Specific Manifestations**:
  - Punctuation used appropriately, without excessive embellishment
  - Aligns with online chat conventions
  - Avoids overuse of multiple exclamation marks, dashes, etc.
- **Priority Assessment**:Appropriate usage > Generally reasonable > Noticeably excessive


**Exemplary Response Example:**
- Replicates natural human reactions with seamless flow, perfectly matching tone to context. Vocabulary and phrasing mirror everyday speech, creating authentic interpersonal exchange.
- Example:
  - User: "My work stress has been overwhelming lately. I feel exhausted every day."
  - "Oh my, that sounds exhausting! Why not take some time off to relax? You really need a break."

**Poor Response Examples:**
- Expressions feel stiff, using formulaic language with insufficient colloquialism; emotional tone is either flat or misplaced
- Example:
  - User: "I've been under immense work pressure lately and feel exhausted every day."
  - "I completely understand how you feel. I suggest adjusting your schedule appropriately and taking necessary breaks."


## Final Judgment Logic

**Evaluation Process:**
1. **Check Direct Elimination Criteria**:First verify if any response triggers immediate disqualification conditions
2. **Analyze Expression Naturalness**:Assess differences  across 7 aspects including Logical Coherence, Emotional Expression, Adherence to Colloquial Norms between the two responses.
3. **Analyze length appropriateness**: Count words to determine if the response burdens the user
4. **Comprehensive judgment**: Combine content quality and length appropriateness to directly determine the final winner and degree of superiority

**Length Penalty Rules:**
- **Under 300 words:** Not affected by length
- **300-480 words:** If the response would have won, its advantage level is reduced by one tier
- **Over 480 words:** Strict penalty; the response must demonstrate significantly superior content quality to win

**Advantage Level Standards:**
- **"+"**: **Slight advantage**
- **"++"**: **Moderate advantage**
- **"+++"**: **Significant advantage**
- **"++++"**: **Clear advantage**
- **"+++++"**: **Overwhelming advantage**

**Comprehensive Evaluation Principles:**
- Content quality is fundamental; length appropriateness serves as an adjusting factor
- Excessively long replies may cause stress to emotionally distressed users and warrant severe penalties
- Final output ratings incorporate all factors

## Judgment Guidelines

- **A clear winner must be selected for Expression Naturalness** (No ties allowed)
- When outputting "Winner & Margin Rating," append it after the winning identifier (A or B), e.g., "A++" indicates A has a slight edge
- **Important:** The final winner must be determined after length penalty adjustment;it cannot be based solely on the Expression Naturalness; However, note that Expression Naturalness remains the primary evaluation factor,with length serving only as an adjustment factor
- **Special Note:** Any response triggering direct elimination criteria is automatically disqualified.

Your response must be valid JSON format without additional comments, adhering to this structure:

{{
  "chain_of_thought_reasoning": "Detailed analysis process:1)Check Direct Elimination Criteria;2)Analyze word counts of both replies; 3)Evaluate differences in Expression Naturalness;4)Directly determine final winner and rationale based on content quality and length appropriateness",
  "disqualification_check": "Check if any reply triggers direct elimination criteria",
  "length_analysis": "Word count statistics and appropriateness assessment for both responses",
  "content_quality_analysis": "Specific comparative analysis ofNaturalness of Expression",
  "overall_winner": "Final winner after comprehensive evaluation and differential scoring"
}}
"""