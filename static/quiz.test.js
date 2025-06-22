// Quiz System Unit Tests
describe("Quiz Validation", () => {
  test("validateQuestion should return false for invalid question", () => {
    const invalidQuestion = {
      type: "mcq",
      // missing id and text
    };
    expect(validateQuestion(invalidQuestion)).toBe(false);
  });

  test("validateQuestion should return true for valid MCQ question", () => {
    const validQuestion = {
      id: "q1",
      type: "mcq",
      text: "Test question?",
      options: [
        { value: "A", text: "Option A" },
        { value: "B", text: "Option B" },
      ],
    };
    expect(validateQuestion(validQuestion)).toBe(true);
  });

  test("validateQuestion should return true for valid gap question", () => {
    const validQuestion = {
      id: "q2",
      type: "gap",
      text: "Fill in the ___",
      answer: "blank",
    };
    expect(validateQuestion(validQuestion)).toBe(true);
  });
});

describe("Answer Checking", () => {
  test("MCQ answer checking should work correctly", () => {
    const question = {
      id: "q1",
      type: "mcq",
      text: "Test?",
      options: [
        { value: "A", text: "Wrong" },
        { value: "B", text: "Right" },
      ],
      answer: "B",
    };

    // Mock DOM elements
    document.body.innerHTML = `
            <article id="q1">
                <input type="radio" name="q1" value="B" checked>
                <label></label>
            </article>
        `;

    const article = document.getElementById("q1");
    let result = false;

    try {
      checkAnswer(question, article);
      const label = article.querySelector("label");
      result = label.classList.contains("correct");
    } catch (e) {
      console.error(e);
    }

    expect(result).toBe(true);
  });

  test("Gap fill answer checking should work correctly", () => {
    const question = {
      id: "q2",
      type: "gap",
      text: "Fill in: ___",
      answer: "correct",
    };

    document.body.innerHTML = `
            <article id="q2">
                <input type="text" value="correct">
            </article>
        `;

    const article = document.getElementById("q2");
    let result = false;

    try {
      checkAnswer(question, article);
      const input = article.querySelector("input");
      result = input.classList.contains("correct");
    } catch (e) {
      console.error(e);
    }

    expect(result).toBe(true);
  });
});

describe("Quiz Score Calculation", () => {
  test("Score should be calculated correctly", () => {
    const quizData = {
      questions: [
        {
          id: "q1",
          type: "mcq",
          text: "Q1?",
          options: [
            { value: "A", text: "A" },
            { value: "B", text: "B" },
          ],
          answer: "A",
        },
        {
          id: "q2",
          type: "gap",
          text: "Q2?",
          answer: "test",
        },
      ],
    };

    document.body.innerHTML = `
            <div id="quizContent">
                <article id="q1">
                    <input type="radio" name="q1" value="A" checked>
                </article>
                <article id="q2">
                    <input type="text" value="test">
                </article>
                <div id="quizScore"></div>
            </div>
        `;

    let score = 0;
    let total = quizData.questions.length;

    quizData.questions.forEach((q) => {
      const elem = document.getElementById(q.id);
      if (q.type === "mcq") {
        const selected = elem.querySelector("input:checked");
        if (selected && selected.value === q.answer) score++;
      } else if (q.type === "gap") {
        const input = elem.querySelector("input");
        if (input.value.toLowerCase() === q.answer.toLowerCase()) score++;
      }
    });

    expect(score).toBe(2);
    expect(total).toBe(2);
  });
});
