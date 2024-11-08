document.addEventListener('DOMContentLoaded', function() {
    function countWords(text) {
        return text.trim().split(/\s+/).filter(word => word.length > 0).length;
    }

    document.getElementById('original').addEventListener('input', function() {
        const wordCount = countWords(this.value);
        document.getElementById('original-word-count').textContent = wordCount;
    });

    document.getElementById('submission').addEventListener('input', function() {
        const wordCount = countWords(this.value);
        document.getElementById('submission-word-count').textContent = wordCount;
    });

    document.getElementById('check-button').addEventListener('click', async function() {
        const button = this;
        const loading = document.getElementById('loading');
        const original = document.getElementById('original').value;
        const submission = document.getElementById('submission').value;

        if (!original || !submission) {
            alert('Please provide both original and submission texts.');
            return;
        }

        button.disabled = true;
        loading.style.display = 'block';
        document.getElementById('results').style.display = 'none';

        try {
            const response = await fetch('/check_plagiarism', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    original: original,
                    submission: submission
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to check plagiarism');
            }
            
            const data = await response.json();
            displayResults(data);
        } catch (error) {
            alert('Error checking plagiarism: ' + error.message);
        } finally {
            button.disabled = false;
            loading.style.display = 'none';
        }
    });

    function displayResults(data) {
        document.getElementById('results').style.display = 'block';
        
        // Display similarity percentage
        const similarityPercentage = (data.overall_similarity * 100).toFixed(2);
        const similarityElement = document.getElementById('similarity');
        similarityElement.textContent = similarityPercentage + '%';
        
        // Set color based on similarity
        if (similarityPercentage > 70) {
            similarityElement.className = 'similarity-badge badge bg-danger';
        } else if (similarityPercentage > 40) {
            similarityElement.className = 'similarity-badge badge bg-warning';
        } else {
            similarityElement.className = 'similarity-badge badge bg-success';
        }

        // Display warning if needed
        const warningElement = document.getElementById('patchwriting-warning');
        warningElement.style.display = data.patchwriting_detected ? 'block' : 'none';
        if (data.patchwriting_detected) {
            warningElement.textContent = '⚠️ Warning: Potential patchwriting or plagiarism detected!';
        }

        // Display text statistics
        document.getElementById('word-count').textContent = data.text_statistics.word_count;
        document.getElementById('sentence-count').textContent = data.text_statistics.sentence_count;
        document.getElementById('unique-words').textContent = data.text_statistics.unique_words;

        // Display similar sentences
        const similarSentencesDiv = document.getElementById('similar-sentences');
        similarSentencesDiv.innerHTML = '';
        
        if (data.similar_sentences.length === 0) {
            similarSentencesDiv.innerHTML = '<p class="text-muted">No significant similarities found.</p>';
        } else {
            data.similar_sentences.forEach(match => {
                const matchElement = document.createElement('div');
                matchElement.className = 'match-card';
                matchElement.innerHTML = `
                    <div class="d-flex justify-content-between mb-2">
                        <strong>Similarity: ${(match.similarity * 100).toFixed(2)}%</strong>
                        <span class="badge bg-secondary">${match.match_type}</span>
                    </div>
                    <div class="mb-2">
                        <small class="text-muted">Submitted Text:</small><br>
                        ${match.submission_sentence}
                    </div>
                    <div>
                        <small class="text-muted">Original Text:</small><br>
                        ${match.original_sentence}
                    </div>
                `;
                similarSentencesDiv.appendChild(matchElement);
            });
        }
    }
}); 