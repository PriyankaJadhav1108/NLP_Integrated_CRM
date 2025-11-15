let NLP_BASE = localStorage.getItem('nlpBase') || 'http://127.0.0.1:8000';
let API_BASE = localStorage.getItem('crmBase') || 'http://127.0.0.1:8000/api/v1';
let API_KEY = localStorage.getItem('apiKey') || '';

async function fetchJSON(url) {
  const r = await fetch(url, { headers: API_KEY ? { 'X-API-Key': API_KEY } : {} });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

function statusBadge(status) {
  const cls = {
    pending: 'status-pending',
    escalated: 'status-escalated',
    resolved: 'status-resolved',
    'in_progress': 'status-pending'
  }[status] || 'status-pending';
  return `<span class="badge ${cls}">${status}</span>`;
}

function interactionCard(i, actions=false) {
  const idShort = (i.interaction_id||'').slice(0,8);
  const sentimentText = (i.sentiment||'').toString().toLowerCase();
  const priorityText = (typeof i.priority === 'number') ? `${i.priority}/5` : (i.priority||'-');
  const confidence = (i.metadata && typeof i.metadata.confidence_score === 'number') ? `${(i.metadata.confidence_score*100).toFixed(0)}%` : '-';
  const processing = (i.metadata && typeof i.metadata.processing_time_ms === 'number') ? `${i.metadata.processing_time_ms} ms` : '-';
  return `
  <div class="interaction" data-id="${i.interaction_id}">
    <div class="interaction-header">
      <div>${idShort}...</div>
      <div class="interaction-actions">
        ${statusBadge(i.status)}
        <button class="btn btn-danger btn-delete" title="Delete this interaction" data-id="${i.interaction_id}">🗑️</button>
      </div>
    </div>
    <div class="interaction-body">
      <div class="meta">
        <div><div class="meta-label">Intent</div><div class="meta-value">${i.intent}</div></div>
        <div><div class="meta-label">Sentiment</div><div class="meta-value">${sentimentText}</div></div>
        <div><div class="meta-label">Priority</div><div class="meta-value">${priorityText}</div></div>
        <div><div class="meta-label">Confidence</div><div class="meta-value">${confidence}</div></div>
        <div><div class="meta-label">Latency</div><div class="meta-value">${processing}</div></div>
        <div><div class="meta-label">Type</div><div class="meta-value">${i.interaction_type}</div></div>
      </div>
      <div class="bubble q"><strong>Customer:</strong> ${i.customer_query}</div>
      <div class="bubble a"><strong>Assistant:</strong> ${i.assistant_response}</div>
    </div>
  </div>`;
}

function timelineItem(i){
  return interactionCard(i, false);
}

async function loadStats(){
  try{
    const r = await fetchJSON(`${API_BASE}/stats`);
    if(!r.success) throw new Error(r.error || 'stats error');
    const s = r.stats || {};
    document.getElementById('totalInteractions').textContent = s.total_interactions || 0;
    document.getElementById('pendingInteractions').textContent = (s.by_status||{}).pending || 0;
    document.getElementById('escalatedInteractions').textContent = (s.by_status||{}).escalated || 0;
    document.getElementById('escalationRate').textContent = (((s.escalation_rate)||0)*100).toFixed(1)+"%";
  }catch(e){
    console.error(e);
  }
}

async function loadTriage(){
  try{
    const r = await fetchJSON(`${API_BASE}/triage?limit=10`);
    const el = document.getElementById('triage');
    if(r.success && r.interactions.length){
      el.innerHTML = r.interactions.map(i=>interactionCard(i,true)).join('');
      // Wire delete buttons
      el.querySelectorAll('.btn-delete').forEach(btn=>{
        btn.addEventListener('click', async (e)=>{
          const id = e.currentTarget.getAttribute('data-id');
          if(!id) return;
          const ok = confirm('Delete this interaction? This cannot be undone.');
          if(!ok) return;
          try{
            const r = await fetch(`${API_BASE}/interactions/${id}`, { method: 'DELETE' });
            const out = await r.json();
            if(!r.ok || !out.success){ throw new Error(out.error||`HTTP ${r.status}`); }
            await refresh();
          }catch(err){
            alert('Delete failed: ' + err.message);
          }
        });
      });
    }else{
      el.innerHTML = '<div class="meta-value">No interactions require triage.</div>';
    }
  }catch(e){
    document.getElementById('triage').innerHTML = '<div class="meta-value">Failed to load triage.</div>';
  }
}

async function loadTimeline(){
  try{
    const r = await fetchJSON(`${API_BASE}/interactions?limit=20`);
    const el = document.getElementById('timeline');
    if(r.success && r.interactions.length){
      el.innerHTML = r.interactions.map(timelineItem).join('');
      // Wire delete buttons in timeline
      el.querySelectorAll('.btn-delete').forEach(btn=>{
        btn.addEventListener('click', async (e)=>{
          const id = e.currentTarget.getAttribute('data-id');
          if(!id) return;
          const ok = confirm('Delete this interaction? This cannot be undone.');
          if(!ok) return;
          try{
            const r = await fetch(`${API_BASE}/interactions/${id}`, { method: 'DELETE' });
            const out = await r.json();
            if(!r.ok || !out.success){ throw new Error(out.error||`HTTP ${r.status}`); }
            await refresh();
          }catch(err){
            alert('Delete failed: ' + err.message);
          }
        });
      });
    }else{
      el.innerHTML = '<div class="meta-value">No recent interactions.</div>';
    }
  }catch(e){
    document.getElementById('timeline').innerHTML = '<div class="meta-value">Failed to load timeline.</div>';
  }
}

async function refresh(){
  await Promise.all([loadStats(), loadTriage(), loadTimeline()]);
}

document.addEventListener('DOMContentLoaded', ()=>{
  try {
  document.getElementById('refreshBtn').addEventListener('click', refresh);
  refresh();
  setInterval(refresh, 30000);

  const askBtn = document.getElementById('askBtn');
  const askInput = document.getElementById('askInput');
  const asrInput = document.getElementById('asrInput');
  const asrBtn = document.getElementById('asrBtn');
  const audioFile = document.getElementById('audioFile');
  const box = document.getElementById('answerBox');
  const txt = document.getElementById('answerText');
  const type = document.getElementById('answerType');
  const conf = document.getElementById('answerConfidence');
  const logBtn = document.getElementById('logBtn');
  const factsBox = document.getElementById('factsBox');
  const factsContent = document.getElementById('factsContent');
  // Quick Insights removed; using NLP Summary card under Ask instead
  const quickInsights = null;
  const qiIntent = null;
  const qiSentiment = null;
  const qiPriority = null;
  const nlpSummarySection = document.getElementById('nlpSummarySection');
  const summaryIntent = document.getElementById('summaryIntent');
  const summarySentiment = document.getElementById('summarySentiment');
  const summaryPriority = document.getElementById('summaryPriority');
  const summaryName = document.getElementById('summaryName');
  const summaryProductId = document.getElementById('summaryProductId');
  const summaryCustomerId = document.getElementById('summaryCustomerId');
  const summaryEmail = document.getElementById('summaryEmail');
  const summaryPhone = document.getElementById('summaryPhone');
  const openSettings = document.getElementById('openSettings');
  const toggleTheme = document.getElementById('toggleTheme');
  const themeFab = document.getElementById('themeFab');
  const settingsModal = document.getElementById('settingsModal');
  const closeSettings = document.getElementById('closeSettings');
  const saveSettings = document.getElementById('saveSettings');
  const nlpBase = document.getElementById('nlpBase');
  const crmBase = document.getElementById('crmBase');
  const apiKey = document.getElementById('apiKey');
  
  // Risk Classification elements
  const riskInput = document.getElementById('riskInput');
  const riskBtn = document.getElementById('riskBtn');
  const riskResults = document.getElementById('riskResults');
  const toxicityStatus = document.getElementById('toxicityStatus');
  const toxicityTerms = document.getElementById('toxicityTerms');
  const complianceStatus = document.getElementById('complianceStatus');
  const complianceTerms = document.getElementById('complianceTerms');
  const escalationWarning = document.getElementById('escalationWarning');
  
  // Enhanced NLP Analysis Function
  async function updateEnhancedNLPSummary(text) {
    try {
      const response = await fetch(`${NLP_BASE}/v1/enhanced-nlp/analyze`, {
        method: 'POST',
        headers: Object.assign({'Content-Type': 'application/json'}, API_KEY ? {'X-API-Key': API_KEY} : {}),
        body: JSON.stringify({ text: text })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.success && data.analysis) {
        const analysis = data.analysis;
        const section = document.getElementById('nlpSummarySection');
        
        if (section) {
          // Update summary
          const summaryEl = document.getElementById('enhancedSummary');
          if (summaryEl) summaryEl.textContent = analysis.summary || '-';
          
          // Update key metrics
          const intentEl = document.getElementById('enhancedIntent');
          const priorityEl = document.getElementById('enhancedPriority');
          const wordCountEl = document.getElementById('enhancedWordCount');
          const languageEl = document.getElementById('enhancedLanguage');
          
          if (intentEl) intentEl.textContent = analysis.intent || '-';
          if (priorityEl) priorityEl.textContent = analysis.priority || '-';
          if (wordCountEl) wordCountEl.textContent = analysis.word_count || '-';
          if (languageEl) languageEl.textContent = analysis.detected_language?.toUpperCase() || '-';
          
          // Update sentiment analysis
          const sentiment = analysis.sentiment;
          if (sentiment && !sentiment.error) {
            const sentimentEl = document.getElementById('enhancedSentiment');
            const polarityEl = document.getElementById('enhancedPolarity');
            const subjectivityEl = document.getElementById('enhancedSubjectivity');
            const toneEl = document.getElementById('enhancedTone');
            
            if (sentimentEl) sentimentEl.textContent = `${sentiment.overall_sentiment} ${sentiment.emoji}`;
            if (polarityEl) polarityEl.textContent = (typeof sentiment.polarity === 'number') ? sentiment.polarity.toFixed(2) : (sentiment.polarity || '-');
            if (subjectivityEl) subjectivityEl.textContent = (typeof sentiment.subjectivity === 'number') ? sentiment.subjectivity.toFixed(2) : (sentiment.subjectivity || '-');
            if (toneEl) toneEl.textContent = sentiment.tone || '-';
          }
          
          // Update entities
          const entitiesEl = document.getElementById('enhancedEntities');
          if (entitiesEl && analysis.entities) {
            if (analysis.entities.length > 0) {
              entitiesEl.innerHTML = analysis.entities.map(entity => 
                `<div class="entity-item"><strong>${entity[0]}</strong> → ${entity[2]}</div>`
              ).join('');
            } else {
              entitiesEl.textContent = 'No significant entities detected';
            }
          }
          
          // Update keywords
          const keywordsEl = document.getElementById('enhancedKeywords');
          if (keywordsEl && analysis.keywords) {
            if (analysis.keywords.length > 0) {
              keywordsEl.innerHTML = analysis.keywords.slice(0, 8).map((keyword, i) => 
                `<div class="keyword-item">${i + 1}. ${keyword}</div>`
              ).join('');
            } else {
              keywordsEl.textContent = 'No significant keywords extracted';
            }
          }
          
          // Update timestamp
          const timestampEl = document.getElementById('enhancedTimestamp');
          if (timestampEl) timestampEl.textContent = analysis.analysis_timestamp || '-';
          
          // Show the section
          section.style.display = 'block';
        }
      } else {
        console.warn('Enhanced NLP analysis failed:', data.error_message);
      }
    } catch (error) {
      console.error('Enhanced NLP analysis error:', error);
    }
  }

  let lastAnswer = null;
  
  // Analyze arbitrary text (e.g., ASR transcript) and update NLP Summary card only
  async function analyzeTextForSummary(text){
    try{
      if(!text || !nlpSummarySection){ return; }
      // Prefer lightweight enhanced NLP endpoint to avoid changing Ask output
      const body = { text };
      const r = await fetch(`${NLP_BASE}/v1/nlp/enhanced`, {
        method: 'POST', 
        headers: Object.assign({'Content-Type': 'application/json'}, API_KEY ? {'X-API-Key': API_KEY} : {}),
        body: JSON.stringify(body)
      });
      const data = await r.json();
      if(!data || data.detail){ throw new Error(data.detail || 'enhanced nlp failed'); }
      const sentObj = data.sentiment || { label: 'neutral', score: 0 };
      if (summaryIntent) summaryIntent.textContent = data.primary_intent || '-';
      if (summarySentiment) summarySentiment.textContent = `${sentObj.label} ${(sentObj.score*100).toFixed ? (sentObj.score*100).toFixed(0) : sentObj.score}%`;
      if (summaryPriority) summaryPriority.textContent = (data.priority != null) ? data.priority : '-';
      const ents = data.extracted_entities || {};
      const pick = (arr)=> (Array.isArray(arr) && arr.length) ? arr[0] : '-';
      if (summaryName) summaryName.textContent = pick(ents.person_entities);
      if (summaryProductId) summaryProductId.textContent = pick(ents.product_ids) || pick(ents.product_entities);
      if (summaryCustomerId) summaryCustomerId.textContent = pick(ents.customer_ids);
      if (summaryEmail) summaryEmail.textContent = pick(ents.emails);
      if (summaryPhone) summaryPhone.textContent = pick(ents.phone_numbers);
      nlpSummarySection.style.display = 'block';
    }catch(e){
      console.warn('NLP summary analyze failed:', e);
    }
  }

  async function callAnswer(){
    const q = askInput.value.trim();
    const asr = asrInput.value.trim();
    
    // Auto-use ASR text if no manual question is typed
    const finalQuery = q || asr;
    
    if(!finalQuery){
      alert('Please type a question or transcribe audio first.');
      askInput.focus();
      return;
    }
    
    const body = { query: finalQuery, asr_text: asr || null };
    const r = await fetch(`${NLP_BASE}/v1/answer`, {
      method: 'POST', headers: Object.assign({'Content-Type': 'application/json'}, API_KEY ? {'X-API-Key': API_KEY} : {}),
      body: JSON.stringify(body)
    });
    const data = await r.json();
    if(!data.success){
      box.style.display = 'block';
      type.textContent = 'error';
      conf.textContent = '';
      txt.textContent = data.error_message || 'Failed to get answer';
      factsBox.style.display = 'none';
      lastAnswer = null;
      return;
    }
    const resp = data.response;
    lastAnswer = { data, req: body };
    box.style.display = 'block';
    type.textContent = resp.response_type;
    conf.textContent = resp.confidence;
    txt.textContent = resp.answer;

    // Call enhanced NLP analysis and update the Enhanced NLP Summary card
    await updateEnhancedNLPSummary(finalQuery);

    // Show definitive facts if available in sources[0].metadata
    const src0 = (resp.sources||[])[0];
    const metaFacts = src0 && src0.metadata;
    if(metaFacts && metaFacts.sku){
      factsBox.style.display = 'block';
      factsContent.innerHTML = `SKU: ${metaFacts.sku}<br/>Name: ${metaFacts.name}<br/>Warranty End: ${metaFacts.warranty_end}<br/>Return Policy: ${metaFacts.return_policy}<br/>Shipping SLA: ${metaFacts.shipping_sla}<br/>Support Tier: ${metaFacts.support_tier}`;
    } else {
      factsBox.style.display = 'none';
      factsContent.innerHTML = '';
    }
  }

  if (askBtn) askBtn.addEventListener('click', callAnswer);
  if (askInput) askInput.addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ callAnswer(); }});

  async function doASR(){
    const f = audioFile.files[0];
    if(!f){ 
      audioFile.click(); 
      return; 
    }
    
    try {
      asrBtn.textContent = 'Transcribing...';
      asrBtn.disabled = true;
      
      const form = new FormData();
      form.append('file', f);
      
      console.log('Sending ASR request to:', `${NLP_BASE}/v1/asr/transcribe`);
      console.log('API Key set:', !!API_KEY);
      
      const r = await fetch(`${NLP_BASE}/v1/asr/transcribe`, {
        method: 'POST',
        headers: API_KEY ? { 'X-API-Key': API_KEY } : {},
        body: form
      });
      
      console.log('ASR response status:', r.status);
      
      if (!r.ok) {
        throw new Error(`HTTP ${r.status}: ${r.statusText}`);
      }
      
      const out = await r.json();
      console.log('ASR response:', out);
      
      if(out && out.success){
        const transcribed = out.text || '';
        asrInput.value = transcribed;
        
        // Auto-fill ALL relevant fields with transcribed audio text
        try { 
          if (typeof riskInput !== 'undefined' && riskInput) { 
            riskInput.value = transcribed; 
          }
          
          // Auto-fill Multilingual AI Negotiator section
          const multilingualAudioFile = document.getElementById('multilingualAudioFile');
          if (multilingualAudioFile && f) {
            // Create a new FileList-like object to simulate file selection
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(f);
            multilingualAudioFile.files = dataTransfer.files;
            
            // Update the file display text
            const fileLabel = multilingualAudioFile.nextElementSibling;
            if (fileLabel) {
              fileLabel.textContent = f.name;
            }
          }
          
          // Auto-detect language and cultural context
          const detectedLanguage = out.language || 'en';
          // Map language code → display name
          const langNames = { en: 'English', hi: 'Hindi', mr: 'Marathi' };
          const detectedLanguageName = langNames[detectedLanguage] || detectedLanguage || 'Unknown';
          // Map language → cultural context (API key + display label)
          const ctxKey = (detectedLanguage === 'en') ? 'western_business' : 'indian_urban';
          const ctxLabel = (ctxKey === 'western_business') ? 'Western Business' : 'Indian Urban';
          
          const languageField = document.getElementById('multilingualPrimaryLanguage');
          const culturalField = document.getElementById('multilingualCulturalContext');
          
          if (languageField) { languageField.value = detectedLanguageName; languageField.dataset.code = detectedLanguage; }
          if (culturalField) { culturalField.value = ctxLabel; culturalField.dataset.key = ctxKey; }
          
        } catch(e) {
          console.warn('Failed to auto-fill fields:', e);
        }
        
        // Clear the manual question field to indicate ASR will be used
        askInput.placeholder = 'ASR text will be used - or type to override';
        console.log('Transcription successful:', transcribed);
        // Immediately analyze transcript for Enhanced NLP Summary card (full analysis)
        updateEnhancedNLPSummary(transcribed);
      } else {
        throw new Error(out.error || 'Transcription failed');
      }
    } catch(e) {
      console.error('ASR error:', e);
      alert('ASR failed: ' + e.message);
    } finally {
      asrBtn.textContent = 'Transcribe Audio';
      asrBtn.disabled = false;
    }
  }
  if (asrBtn) asrBtn.addEventListener('click', doASR);

  async function logToCrm(){
    if(!lastAnswer){ return; }
    try {
      if (logBtn) { logBtn.textContent = 'Logging...'; logBtn.disabled = true; }
    } catch(_){}
    const resp = lastAnswer.data.response;
    
    // Get risk classification data if available
    let riskData = null;
    try {
      const riskResponse = await fetch(`${NLP_BASE}/v1/risk/classify`, {
        method: 'POST', 
        headers: Object.assign({'Content-Type': 'application/json'}, API_KEY ? {'X-API-Key': API_KEY} : {}),
        body: JSON.stringify({ text: lastAnswer.req.query })
      });
      if(riskResponse.ok){
        riskData = await riskResponse.json();
      }
    } catch(e){
      console.warn('Risk classification failed during logging:', e);
    }
    
    const payload = {
      customer_query: lastAnswer.req.query,
      assistant_response: resp.answer,
      response_type: resp.response_type,
      confidence: resp.confidence,
      session_id: lastAnswer.data.session_id || 'ui_session',
      customer_id: 'ui_customer',
      priority: resp.escalation_required ? 3 : 2,
      tags: resp.suggested_actions || [],
      sources: resp.sources || [],
      escalation_required: !!resp.escalation_required,
      escalation_reason: resp.escalation_reason || null,
      processing_time_ms: lastAnswer.data.processing_time_ms || null,
      retrieved_docs_count: (resp.sources||[]).length,
      confidence_score: 0.8,
      // Risk classification data
      risk_classification: riskData ? {
        is_toxic_flagged: riskData.is_toxic_flagged,
        is_compliance_flagged: riskData.is_compliance_flagged,
        toxicity_terms: riskData.toxicity_terms || [],
        compliance_terms: riskData.compliance_terms || []
      } : null
    };
    try {
      const r = await fetch(`${API_BASE}/interactions/log`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const out = await r.json();
      if(out.success){
        // Optimistic UI: prepend to timeline immediately
        try {
          const tl = document.getElementById('timeline');
          if (tl) {
            const interaction = {
              interaction_id: out.interaction_id || 'ui_pending',
              status: payload.escalation_required ? 'escalated' : 'pending',
              interaction_type: payload.response_type || 'query',
              intent: (resp.intent || 'general')
                || (lastAnswer?.data?.response?.intent) || 'general',
              sentiment: (lastAnswer?.data?.response?.sentiment) || 'unknown',
              priority: payload.priority || 2,
              customer_query: payload.customer_query,
              assistant_response: payload.assistant_response
            };
            const card = interactionCard(interaction, false);
            tl.innerHTML = card + tl.innerHTML;
          }
        } catch(_){ /* ignore optimistic failures */ }
        await refresh();
        try { if (logBtn) { logBtn.textContent = 'Logged'; } } catch(_){}
      } else {
        alert('Failed to log: ' + (out.error||'unknown error'));
        try { if (logBtn) { logBtn.textContent = 'Log to CRM'; } } catch(_){}
      }
    } catch(e){
      alert('Failed to log: ' + e.message);
      try { if (logBtn) { logBtn.textContent = 'Log to CRM'; } } catch(_){}
    } finally {
      try { if (logBtn) { logBtn.disabled = false; } } catch(_){}
    }
  }

  if (logBtn) logBtn.addEventListener('click', logToCrm);
  if (quickInsights) quickInsights.addEventListener('click', ()=>{
    const nlpSection = document.getElementById('nlpSection');
    if (nlpSection) nlpSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
  // Theme toggle
  try {
    const applyThemeClasses = (enableDark) => {
      const body = document.body;
      if (enableDark) { body.classList.add('dark'); } else { body.classList.remove('dark'); }
      // Apply .dark variant to key components for better contrast
      document.querySelectorAll('.card').forEach(el=> enableDark ? el.classList.add('dark') : el.classList.remove('dark'));
      document.querySelectorAll('.card-header').forEach(el=> enableDark ? el.classList.add('dark') : el.classList.remove('dark'));
      document.querySelectorAll('.input').forEach(el=> enableDark ? el.classList.add('dark') : el.classList.remove('dark'));
      document.querySelectorAll('.answer-box').forEach(el=> enableDark ? el.classList.add('dark') : el.classList.remove('dark'));
      document.querySelectorAll('.answer-text').forEach(el=> enableDark ? el.classList.add('dark') : el.classList.remove('dark'));
      document.querySelectorAll('.pill').forEach(el=> enableDark ? el.classList.add('dark') : el.classList.remove('dark'));
      document.querySelectorAll('.facts').forEach(el=> enableDark ? el.classList.add('dark') : el.classList.remove('dark'));
      document.querySelectorAll('.interaction').forEach(el=> enableDark ? el.classList.add('dark') : el.classList.remove('dark'));
      document.querySelectorAll('.interaction-header').forEach(el=> enableDark ? el.classList.add('dark') : el.classList.remove('dark'));
      document.querySelectorAll('.meta-label').forEach(el=> enableDark ? el.classList.add('dark') : el.classList.remove('dark'));
      document.querySelectorAll('.meta-value').forEach(el=> enableDark ? el.classList.add('dark') : el.classList.remove('dark'));
      document.querySelectorAll('.q').forEach(el=> enableDark ? el.classList.add('dark') : el.classList.remove('dark'));
      document.querySelectorAll('.a').forEach(el=> enableDark ? el.classList.add('dark') : el.classList.remove('dark'));
      document.querySelectorAll('.risk-results').forEach(el=> enableDark ? el.classList.add('dark') : el.classList.remove('dark'));
      document.querySelectorAll('.risk-indicator').forEach(el=> enableDark ? el.classList.add('dark') : el.classList.remove('dark'));
      document.querySelectorAll('.message-content').forEach(el=> enableDark ? el.classList.add('dark') : el.classList.remove('dark'));
      const header = document.querySelector('.header');
      if (header) header.classList.toggle('dark', enableDark);
    };
    const savedTheme = localStorage.getItem('theme') || 'light';
    applyThemeClasses(savedTheme === 'dark');
    const handleThemeToggle = ()=>{
      const newTheme = (localStorage.getItem('theme') === 'dark') ? 'light' : 'dark';
      localStorage.setItem('theme', newTheme);
      applyThemeClasses(newTheme === 'dark');
    };
    if (toggleTheme) toggleTheme.addEventListener('click', handleThemeToggle);
    if (themeFab) themeFab.addEventListener('click', handleThemeToggle);
  } catch(e) { console.warn('Theme toggle init failed:', e); }

  // Risk Classification functionality
  async function analyzeRisk(){
    const text = riskInput.value.trim();
    if(!text){
      alert('Please enter text to analyze for risks.');
      riskInput.focus();
      return;
    }
    
    try {
      riskBtn.textContent = 'Analyzing...';
      riskBtn.disabled = true;
      
      const body = { text: text };
      const r = await fetch(`${NLP_BASE}/v1/risk/classify`, {
        method: 'POST', 
        headers: Object.assign({'Content-Type': 'application/json'}, API_KEY ? {'X-API-Key': API_KEY} : {}),
        body: JSON.stringify(body)
      });
      const data = await r.json();
      
      if(!r.ok){
        throw new Error(data.detail || `HTTP ${r.status}`);
      }
      
      // Display results
      displayRiskResults(data);
      // Mark as analyzed on button
      riskBtn.textContent = 'Analyzed';
      riskBtn.disabled = false;
      
    } catch(e){
      console.error('Risk analysis failed:', e);
      alert('Risk analysis failed: ' + e.message);
    } finally {
      // keep 'Analyzed' state set above on success; on error reset
      if (riskBtn && riskBtn.textContent !== 'Analyzed'){
        riskBtn.textContent = 'Analyze Risk';
      }
      if (riskBtn && riskBtn.disabled && riskBtn.textContent === 'Analyze Risk'){
        riskBtn.disabled = false;
      }
    }
  }

  function displayRiskResults(data){
    riskResults.style.display = 'block';
    
    // Update toxicity status
    if(data.is_toxic_flagged){
      toxicityStatus.textContent = 'FLAGGED';
      toxicityStatus.className = 'risk-status danger';
      toxicityTerms.innerHTML = data.toxicity_terms.map(term => `<span>${term}</span>`).join('');
    } else {
      toxicityStatus.textContent = 'SAFE';
      toxicityStatus.className = 'risk-status safe';
      toxicityTerms.innerHTML = '';
    }
    
    // Update compliance status
    if(data.is_compliance_flagged){
      complianceStatus.textContent = 'FLAGGED';
      complianceStatus.className = 'risk-status danger';
      complianceTerms.innerHTML = data.compliance_terms.map(term => `<span>${term}</span>`).join('');
    } else {
      complianceStatus.textContent = 'SAFE';
      complianceStatus.className = 'risk-status safe';
      complianceTerms.innerHTML = '';
    }
    
    // Show escalation warning if any risk is flagged
    if(data.is_toxic_flagged || data.is_compliance_flagged){
      escalationWarning.style.display = 'flex';
    } else {
      escalationWarning.style.display = 'none';
    }
  }

  // Event listeners for risk classification
  if (riskBtn) riskBtn.addEventListener('click', analyzeRisk);
  if (riskInput) riskInput.addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ analyzeRisk(); }});



  // Settings modal
  function refreshSettingsUI(){
    nlpBase.value = NLP_BASE;
    crmBase.value = API_BASE;
    apiKey.value = API_KEY;
  }
  if (openSettings) openSettings.addEventListener('click', ()=>{ refreshSettingsUI(); settingsModal.style.display='flex'; });
  if (closeSettings) closeSettings.addEventListener('click', ()=>{ settingsModal.style.display='none'; });
  if (saveSettings) saveSettings.addEventListener('click', ()=>{
    NLP_BASE = nlpBase.value.trim() || NLP_BASE;
    API_BASE = crmBase.value.trim() || API_BASE;
    API_KEY = apiKey.value.trim();
    localStorage.setItem('nlpBase', NLP_BASE);
    localStorage.setItem('crmBase', API_BASE);
    localStorage.setItem('apiKey', API_KEY);
    settingsModal.style.display='none';
    refresh();
  });

  // Multilingual Negotiator functionality
  // Multilingual negotiator section removed
    // Expose a few helpers for debugging
    window.doASR = doASR;
    window.analyzeRisk = analyzeRisk;
    window.callAnswer = callAnswer;
    console.log('[NLPCRM] UI initialized successfully');
  } catch (e) {
    console.error('[NLPCRM] UI initialization failed:', e);
  }
});


