document.addEventListener('DOMContentLoaded', function () {
    const submitButton = document.getElementById('submitQuiz');
    const quizContent = document.getElementById('quizContent');
    const quizTitle = document.getElementById('quizTitle');

    // Load quiz data from JSON
    const quizData = JSON.parse(document.querySelector('script.quiz-json').textContent);
    
    // Set quiz title
    quizTitle.textContent = quizData.quizTitle;

    // Render questions
    quizData.questions.forEach(question => {
        const article = document.createElement('article');
        article.className = 'q';
        article.dataset.type = question.type;
        article.id = question.id;

        // Create question header
        const h3 = document.createElement('h3');
        h3.textContent = question.text;
        article.appendChild(h3);

        // Create question content based on type
        if (question.type === 'mcq' || question.type === 'tf') {
            const ul = document.createElement('ul');
            question.options.forEach(option => {
                const li = document.createElement('li');
                const label = document.createElement('label');
                const input = document.createElement('input');
                input.type = 'radio';
                input.name = question.id;
                input.value = option.value;
                
                const span = document.createElement('span');
                span.textContent = option.text;
                
                label.appendChild(input);
                label.appendChild(document.createTextNode(' '));
                label.appendChild(span);
                li.appendChild(label);
                ul.appendChild(li);
            });
            article.appendChild(ul);
        } else if (question.type === 'gap') {
            const input = document.createElement('input');
            input.type = 'text';
            input.name = question.id;
            input.placeholder = 'Your answer';
            article.appendChild(input);
        } else if (question.type === 'short') {
            const input = document.createElement('input');
            input.type = 'text';
            input.name = question.id;
            input.placeholder = question.placeholder || 'Your answer';
            article.appendChild(input);
        }

        // Add hint if exists
        if (question.hint) {
            const hint = document.createElement('p');
            hint.className = 'hint';
            hint.textContent = question.hint;
            hint.hidden = true;
            article.appendChild(hint);
        }

        // Add feedback message if exists
        if (question.feedback) {
            const feedback = document.createElement('p');
            feedback.className = 'feedback-message';
            feedback.textContent = question.feedback;
            feedback.hidden = true;
            article.appendChild(feedback);
        }

        quizContent.appendChild(article);
    });

    submitButton.addEventListener('click', function () {
        let score = 0;
        let totalQuestions = 0;

        quizData.questions.forEach(question => {
            const questionElement = document.getElementById(question.id);
            const hint = questionElement.querySelector('.hint');
            const feedbackMessage = questionElement.querySelector('.feedback-message');

            if (hint) hint.hidden = false;
            if (feedbackMessage) feedbackMessage.hidden = false;
            
            totalQuestions++;
            let isCorrect = false;

            if (question.type === 'mcq' || question.type === 'tf') {
                const selectedOption = questionElement.querySelector('input[name="' + question.id + '"]:checked');
                if (selectedOption) {
                    const userAnswer = selectedOption.value;
                    const correctAnswer = question.answer;
                    const label = selectedOption.closest('label');
                    if (userAnswer === correctAnswer) {
                        isCorrect = true;
                        if (label) label.classList.add('correct');
                    } else {
                        if (label) label.classList.add('wrong');
                        const correctOptionInput = questionElement.querySelector('input[name="' + question.id + '"][value="' + correctAnswer + '"]');
                        if (correctOptionInput) {
                            correctOptionInput.closest('label').classList.add('correct');
                        }
                    }
                } else {
                    const correctOptionInput = questionElement.querySelector('input[name="' + question.id + '"][value="' + question.answer + '"]');
                    if (correctOptionInput) {
                        correctOptionInput.closest('label').classList.add('correct');
                        const hintElement = questionElement.querySelector('.hint');
                        if(hintElement) {
                            const noAnswerMsg = document.createElement('p');
                            noAnswerMsg.textContent = 'No answer selected. Correct answer highlighted.';
                            noAnswerMsg.style.color = 'var(--wrong)';
                            hintElement.parentNode.insertBefore(noAnswerMsg, hintElement.nextSibling);
                        }
                    }
                }
            } else if (question.type === 'gap') {
                const inputElement = questionElement.querySelector('input[type="text"]');
                const userAnswer = inputElement.value.trim().toLowerCase();
                const correctAnswer = question.answer.toLowerCase();
                if (userAnswer === correctAnswer) {
                    isCorrect = true;
                    inputElement.classList.add('correct');
                    inputElement.classList.remove('wrong');
                } else {
                    inputElement.classList.add('wrong');
                    inputElement.classList.remove('correct');
                    const correctAnswerDisplay = document.createElement('p');
                    correctAnswerDisplay.innerHTML = `Correct answer: <span class="correct-answer">${question.answer}</span>`;
                    if (!inputElement.nextElementSibling || !inputElement.nextElementSibling.classList.contains('correct-answer-display')) {
                        inputElement.insertAdjacentElement('afterend', correctAnswerDisplay);
                        correctAnswerDisplay.classList.add('correct-answer-display');
                    }
                }
            } else if (question.type === 'short') {
                const inputElement = questionElement.querySelector('input[type="text"]');
                const userAnswer = inputElement.value.trim().toLowerCase();
                
                if (question.keywords) {
                    let keywordMatchCount = 0;
                    question.keywords.forEach(keyword => {
                        if (userAnswer.includes(keyword.toLowerCase())) {
                            keywordMatchCount++;
                        }
                    });
                    if (keywordMatchCount >= 2 && userAnswer.length > 10) {
                        isCorrect = true;
                        inputElement.classList.add('correct');
                        inputElement.classList.remove('wrong');
                    } else if (userAnswer.length > 0) {
                        inputElement.classList.add('wrong');
                        inputElement.classList.remove('correct');
                    } else {
                        inputElement.classList.add('wrong');
                    }
                } else {
                    if (userAnswer.length > 0) {
                        isCorrect = true;
                        inputElement.style.borderColor = 'var(--accent)';
                    }
                }
            }
            if(isCorrect) score++;
        });
        
        const quizScoreDiv = document.getElementById('quizScore');
        quizScoreDiv.textContent = `Your Score: ${score} out of ${totalQuestions}`;
        quizScoreDiv.hidden = false;
        submitButton.disabled = true;
        submitButton.textContent = 'Results Shown';
        submitButton.style.backgroundColor = 'var(--hint-fg)';
    });
}); 