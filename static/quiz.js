document.addEventListener("DOMContentLoaded", function () {
  const submitButton = document.getElementById("submitQuiz");
  const quizContent = document.getElementById("quizContent");
  const quizTitle = document.getElementById("quizTitle");

  // Load quiz data from JSON
  const quizData = JSON.parse(
    document.querySelector("script.quiz-json").textContent
  );

  // Set quiz title
  quizTitle.textContent = quizData.quizTitle;

  // Render questions
  quizData.questions.forEach((question) => {
    const article = document.createElement("article");
    article.className = "q";
    article.dataset.type = question.type;
    article.id = question.id;

    // Create question header
    const h3 = document.createElement("h3");
    h3.textContent = question.text;
    article.appendChild(h3);

    // Create question content based on type
    if (question.type === "mcq" || question.type === "tf") {
      const ul = document.createElement("ul");
      question.options.forEach((option) => {
        const li = document.createElement("li");
        const label = document.createElement("label");
        const input = document.createElement("input");
        input.type = "radio";
        input.name = question.id;
        input.value = option.value;

        const span = document.createElement("span");
        span.textContent = option.text;

        label.appendChild(input);
        label.appendChild(document.createTextNode(" "));
        label.appendChild(span);
        li.appendChild(label);
        ul.appendChild(li);
      });
      article.appendChild(ul);
    } else if (question.type === "gap") {
      const input = document.createElement("input");
      input.type = "text";
      input.name = question.id;
      input.placeholder = "Your answer";
      article.appendChild(input);
    } else if (question.type === "short") {
      const input = document.createElement("input");
      input.type = "text";
      input.name = question.id;
      input.placeholder = question.placeholder || "Your answer";
      article.appendChild(input);
    }

    // Add hint if exists
    if (question.hint) {
      const hint = document.createElement("p");
      hint.className = "hint";
      hint.textContent = question.hint;
      hint.hidden = true;
      article.appendChild(hint);
    }

    // Add feedback message if exists
    if (question.feedbackMessage) {
      const feedback = document.createElement("p");
      feedback.className = "feedback-message";
      feedback.textContent = question.feedbackMessage;
      feedback.hidden = true;
      article.appendChild(feedback);
    }

    // Add check button for each question
    const checkButton = document.createElement("button");
    checkButton.textContent = "Check Answer";
    checkButton.className = "check-button";
    checkButton.addEventListener("click", function () {
      checkAnswer(question, article);
    });
    article.appendChild(checkButton);

    quizContent.appendChild(article);
  });

  function checkAnswer(question, questionElement) {
    const hint = questionElement.querySelector(".hint");
    const feedbackMessage = questionElement.querySelector(".feedback-message");
    let isCorrect = false;

    // Get correct answer from question or answers object
    const correctAnswer = question.answer || quizData.answers[question.id];
    if (!correctAnswer) {
      console.error("No correct answer found for question:", question.id);
      return;
    }

    if (question.type === "mcq" || question.type === "tf") {
      // Clear previous highlighting
      const allLabels = questionElement.querySelectorAll("label");
      allLabels.forEach((label) => {
        label.classList.remove("correct", "wrong");
      });

      const selectedOption = questionElement.querySelector(
        `input[name="${question.id}"]:checked`
      );

      // Always highlight the correct answer
      const correctLabel = questionElement
        .querySelector(`input[value="${correctAnswer}"]`)
        ?.closest("label");
      if (correctLabel) {
        correctLabel.classList.add("correct");
      }

      if (selectedOption) {
        const userAnswer = selectedOption.value;
        const selectedLabel = selectedOption.closest("label");

        // For true/false questions, do case-insensitive comparison
        isCorrect =
          question.type === "tf"
            ? userAnswer.toLowerCase() ===
              correctAnswer.toString().toLowerCase()
            : userAnswer === correctAnswer;

        if (!isCorrect && selectedLabel) {
          selectedLabel.classList.add("wrong");
        }
      }
    } else if (question.type === "gap") {
      const inputElement = questionElement.querySelector('input[type="text"]');
      if (!inputElement) return;

      const userAnswer = inputElement.value.trim().toLowerCase();
      const correctAnswerLower = correctAnswer.toString().toLowerCase();

      // Remove existing correct answer display
      const existingDisplay = questionElement.querySelector(
        ".correct-answer-display"
      );
      if (existingDisplay) {
        existingDisplay.remove();
      }

      isCorrect = userAnswer === correctAnswerLower;
      inputElement.classList.remove("correct", "wrong");
      inputElement.classList.add(isCorrect ? "correct" : "wrong");

      if (!isCorrect && userAnswer.length > 0) {
        const correctAnswerDisplay = document.createElement("p");
        correctAnswerDisplay.innerHTML = `Correct answer: <span class="correct-answer">${correctAnswer}</span>`;
        correctAnswerDisplay.classList.add("correct-answer-display");
        inputElement.insertAdjacentElement("afterend", correctAnswerDisplay);
      }
    } else if (question.type === "short") {
      const inputElement = questionElement.querySelector('input[type="text"]');
      if (!inputElement) return;

      const userAnswer = inputElement.value.trim().toLowerCase();
      inputElement.classList.remove("correct", "wrong");

      if (question.keywords) {
        let keywordMatchCount = 0;
        question.keywords.forEach((keyword) => {
          if (userAnswer.includes(keyword.toLowerCase())) {
            keywordMatchCount++;
          }
        });
        isCorrect = keywordMatchCount >= 2 && userAnswer.length > 10;
      } else {
        isCorrect = userAnswer.length > 0;
      }
      inputElement.classList.add(isCorrect ? "correct" : "wrong");
    }

    // Show/hide hint and feedback
    if (hint) {
      hint.hidden = isCorrect;
    }
    if (feedbackMessage) {
      feedbackMessage.hidden = isCorrect;
    }

    // Update check button
    const checkButton = questionElement.querySelector(".check-button");
    if (checkButton) {
      checkButton.textContent = isCorrect ? "Correct! ✓" : "Try Again ×";
      checkButton.classList.remove("correct", "wrong");
      checkButton.classList.add(isCorrect ? "correct" : "wrong");
    }
  }

  submitButton.addEventListener("click", function () {
    let score = 0;
    let totalQuestions = 0;

    quizData.questions.forEach((question) => {
      const questionElement = document.getElementById(question.id);
      if (!questionElement) return;

      totalQuestions++;
      const correctAnswer = question.answer || quizData.answers[question.id];
      if (!correctAnswer) return;

      if (question.type === "mcq" || question.type === "tf") {
        const selectedOption = questionElement.querySelector(
          `input[name="${question.id}"]:checked`
        );
        if (selectedOption) {
          const isCorrect =
            question.type === "tf"
              ? selectedOption.value.toLowerCase() ===
                correctAnswer.toString().toLowerCase()
              : selectedOption.value === correctAnswer;
          if (isCorrect) score++;
        }
      } else if (question.type === "gap") {
        const inputElement =
          questionElement.querySelector('input[type="text"]');
        if (inputElement) {
          const userAnswer = inputElement.value.trim().toLowerCase();
          if (userAnswer === correctAnswer.toString().toLowerCase()) {
            score++;
          }
        }
      } else if (question.type === "short") {
        const inputElement =
          questionElement.querySelector('input[type="text"]');
        if (inputElement) {
          const userAnswer = inputElement.value.trim().toLowerCase();
          if (question.keywords) {
            let keywordMatchCount = 0;
            question.keywords.forEach((keyword) => {
              if (userAnswer.includes(keyword.toLowerCase())) {
                keywordMatchCount++;
              }
            });
            if (keywordMatchCount >= 2 && userAnswer.length > 10) {
              score++;
            }
          } else if (userAnswer.length > 0) {
            score++;
          }
        }
      }
    });

    const quizScoreDiv = document.getElementById("quizScore");
    quizScoreDiv.textContent = `Your Score: ${score} out of ${totalQuestions}`;
    quizScoreDiv.hidden = false;
    submitButton.disabled = true;
    submitButton.textContent = "Quiz Completed";
    submitButton.style.backgroundColor = "var(--hint-fg)";

    // Show all correct answers
    quizData.questions.forEach((question) => {
      const element = document.getElementById(question.id);
      if (element) {
        checkAnswer(question, element);
      }
    });
  });
});
