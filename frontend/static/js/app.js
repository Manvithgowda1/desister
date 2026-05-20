document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const micBtn = document.getElementById('mic-btn');
    const sosIndicator = document.getElementById('sos-indicator');
    const connectionStatus = document.getElementById('connection-status');

    let isRecording = false;
    let recognition = null;
    let isAppOnline = false;

    refreshConnectionStatus();

    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.lang = 'en-US';
        recognition.maxAlternatives = 3;

        recognition.onstart = () => {
            isRecording = true;
            micBtn.classList.add('recording');
            messageInput.placeholder = "Listening...";
        };

        recognition.onresult = (event) => {
            let finalTranscript = '';
            let interimTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }
            // Show interim results in the input field
            if (interimTranscript) {
                messageInput.value = interimTranscript;
            }
            if (finalTranscript) {
                messageInput.value = finalTranscript;
                sendMessage(true);
            }
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error', event.error);
            stopRecording();
        };

        recognition.onend = () => {
            stopRecording();
        };
    } else {
        micBtn.style.display = 'none';
        console.warn("Speech Recognition API not supported in this browser.");
    }

    function stopRecording() {
        isRecording = false;
        micBtn.classList.remove('recording');
        messageInput.placeholder = "Type your emergency query here...";
    }

    micBtn.addEventListener('click', () => {
        if (!recognition) {
            alert('Speech recognition is not supported in this browser. Please use Chrome or Edge.');
            return;
        }

        if (isRecording) {
            recognition.stop();
        } else {
            window.speechSynthesis.cancel();
            try {
                recognition.start();
            } catch (e) {
                console.error('Could not start speech recognition:', e);
                stopRecording();
            }
        }
    });

    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage(false);
        }
    });

    sendBtn.addEventListener('click', () => sendMessage(false));

    async function refreshConnectionStatus() {
        if (!connectionStatus) return;
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            isAppOnline = !!data.online;
            connectionStatus.classList.toggle('online', isAppOnline);
            connectionStatus.classList.toggle('offline', !isAppOnline);
            const label = connectionStatus.querySelector('span');
            if (label) {
                label.textContent = isAppOnline
                    ? 'Online — visuals enabled'
                    : 'Offline — text only';
            }
        } catch {
            isAppOnline = false;
            connectionStatus.classList.add('offline');
            connectionStatus.classList.remove('online');
            const label = connectionStatus.querySelector('span');
            if (label) label.textContent = 'Offline — text only';
        }
    }

    async function sendMessage(isVoice = false) {
        const text = messageInput.value.trim();
        if (!text) return;

        messageInput.value = '';
        messageInput.style.height = 'auto';

        window.speechSynthesis.cancel();
        sosIndicator.classList.add('hidden');

        appendMessage(text, 'user');

        const typingId = appendTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            const data = await response.json();
            removeElement(typingId);

            if (data.error) {
                appendMessage("Error: " + data.error, 'system');
                return;
            }

            if (data.is_emergency) {
                sosIndicator.classList.remove('hidden');
            }

            isAppOnline = !!data.online;
            refreshConnectionStatus();

            appendMessage(data.response, 'system', {
                images: data.images || [],
                visualGuide: data.visual_guide_available,
                offlineTextOnly: data.offline_text_only,
                urgency: data.urgency || {},
            });

            // Only speak aloud when the user used voice input
            if (isVoice) {
                speakText(data.response);
            }

        } catch (error) {
            console.error("Error communicating with backend:", error);
            removeElement(typingId);
            appendMessage("Sorry, I couldn't reach the backend server. Please try again.", 'system');
        }
    }

    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function appendMessage(text, sender, extras = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        const avatarIcon = sender === 'user' ? 'fa-user' : 'fa-robot';

        const formattedText = escapeHtml(text)
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');

        let imagesHtml = '';
        const images = extras.images || [];
        if (images.length > 0 && extras.visualGuide) {
            imagesHtml = '<div class="visual-guide">' +
                images.map((img) => {
                    const imgUrl = escapeHtml(img.src || img.url);
                    const cap = escapeHtml(img.caption || 'Emergency visual guide');
                    const orig = escapeHtml(img.url);
                    return `
                    <figure class="guide-image">
                        <div class="guide-image-frame">
                            <img src="${imgUrl}" alt="${cap}" loading="lazy" decoding="async"
                                 onerror="this.style.display='none'; var e=this.parentNode.querySelector('.img-load-error'); if(e) e.style.display='block';" />
                            <p class="img-load-error" style="display:none">Image could not be loaded. <a href="${orig}" target="_blank" rel="noopener">Open original</a></p>
                        </div>
                        <figcaption>${cap}</figcaption>
                    </figure>
                `;
                }).join('') +
                '</div>';
        } else if (extras.offlineTextOnly) {
            imagesHtml = '<p class="offline-visual-note"><i class="fa-solid fa-wifi-slash"></i> Offline — visual guide unavailable (text only).</p>';
        }

        let urgencyHtml = '';
        if (extras.urgency && extras.urgency.level) {
            urgencyHtml = `
                <div class="urgency-badge" style="background-color: ${extras.urgency.color}15; color: ${extras.urgency.color}; border: 1px solid ${extras.urgency.color}40; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; display: inline-flex; align-items: center; gap: 6px; margin-bottom: 8px;">
                    ${extras.urgency.icon} ${extras.urgency.label}
                </div>
            `;
        }

        messageDiv.innerHTML = `
            <div class="avatar"><i class="fa-solid ${avatarIcon}"></i></div>
            <div class="message-content">
                ${urgencyHtml}
                <p>${formattedText}</p>
                ${imagesHtml}
            </div>
        `;

        chatBox.appendChild(messageDiv);
        scrollToBottom();
    }

    function appendTypingIndicator() {
        const id = 'typing-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.id = id;
        messageDiv.className = `message system`;
        messageDiv.innerHTML = `
            <div class="avatar"><i class="fa-solid fa-robot"></i></div>
            <div class="message-content">
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        chatBox.appendChild(messageDiv);
        scrollToBottom();
        return id;
    }

    function removeElement(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function speakText(text) {
        if (!('speechSynthesis' in window)) return;
        window.speechSynthesis.cancel(); // Stop any ongoing speech
        const cleanText = text.replace(/[*#]/g, '').replace(/\n/g, '. ');
        const utterance = new SpeechSynthesisUtterance(cleanText);
        utterance.rate = 0.95;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        // Prefer a clear English voice
        const voices = window.speechSynthesis.getVoices();
        const preferred = voices.find(v => v.lang.startsWith('en') && v.name.includes('Google'));
        if (preferred) utterance.voice = preferred;
        window.speechSynthesis.speak(utterance);
    }

    // Preload voices (Chrome loads them async)
    if ('speechSynthesis' in window) {
        window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
    }
});
