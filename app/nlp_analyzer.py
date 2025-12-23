from typing import List, Dict, Any, Optional, Tuple
import re
from collections import Counter
import math
import os
import json
from datetime import datetime, timedelta
import unicodedata
import difflib
from dotenv import load_dotenv, find_dotenv, dotenv_values

# Load environment variables, ensuring .env overrides any existing env vars (fix invalid key precedence)
load_dotenv(find_dotenv(), override=True)

class NLPAnalyzer:
    """Enhanced NLP analyzer with summarization, action item extraction, and keyword extraction."""

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        self.model_name = model_name
        self.summarizer = None
        
        # Initialize OpenAI client for action item extraction and summarization
        self.openai_client = None
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_summary_model = os.getenv('OPENAI_SUMMARY_MODEL', 'gpt-4o-mini')
        # Allow separate model for action extraction; default to summary model (often gpt-4o-mini)
        self.openai_action_model = os.getenv('OPENAI_ACTION_MODEL', self.openai_summary_model)
        if self.openai_api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                print(f"OpenAI client initialized. Summary model: {self.openai_summary_model}, Action model: {self.openai_action_model}")
            except ImportError:
                print("OpenAI library not installed. Falling back to rule-based extraction.")
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}. Falling back to rule-based extraction.")
        
        # Action item patterns for rule-based extraction
        self.action_patterns = [
            # Future tense patterns
            r'\b(will|shall|going to|need to|have to|must|should)\s+([^.!?]{5,50})',
            # Task keywords
            r'\b(todo|task|action item|follow up|next step)[:]*\s*([^.!?]+)',
            # Assignment patterns
            r'\b([A-Z][a-z]+)\s+(will|should|needs to|must)\s+([^.!?]{5,50})',
            # Meeting outcomes
            r'\b(decided|agreed|committed)\s+(?:to|that)\s+([^.!?]{5,50})',
            # Deadline patterns
            r'\b(complete|finish|deliver|submit)\s+(.+?)\s+by\s+([^.!?]+)',
        ]
        
        # Common stop words for keyword extraction
        self.stop_words = set([
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'with', 'to', 'for', 'of', 'as', 'from', 'by', 'that', 'this',
            'it', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'can', 'need', 'we', 'you', 'they', 'he', 'she', 'i',
            'me', 'us', 'them', 'him', 'her', 'my', 'your', 'our', 'their', 'his',
            'its', 'what', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
            'both', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'now', 'also', 'about',
            'okay', 'ok', 'yeah', 'yes', 'no', 'um', 'uh', 'like', 'well'
        ])

    def _load_summarizer(self):
        # Local LLM summarizer removed; using OpenAI only.
        return

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        """Generate a summary of the text using OpenAI only."""
        if not text:
            return ''
        if self.openai_client:
            try:
                return self._summarize_openai(text, target_words=max(80, min(220, max_length)))
            except Exception as e:
                print(f"OpenAI summarization failed: {e}. Returning trimmed text.")
                return text[:max_length*5]
        # If OpenAI client not initialized, return trimmed text
        return text[:max_length*5]

    def _summarize_openai(self, text: str, target_words: int = 150) -> str:
        """Summarize using OpenAI with chunking + synthesis."""
        model = os.getenv('OPENAI_SUMMARY_MODEL', self.openai_summary_model)
        
        # Calculate transcript word count and ensure summary is < 50% of original
        transcript_word_count = len(text.split())
        max_summary_words = int(transcript_word_count * 0.45)  # 45% to ensure we stay under 50%
        
        # Use the smaller of the requested target_words or the 45% limit
        # Also ensure a minimum of 50 words and maximum of 500 words for quality
        target_words = max(50, min(target_words, max_summary_words, 500))
        
        print(f"Transcript: {transcript_word_count} words | Target summary: {target_words} words ({(target_words/transcript_word_count)*100:.1f}%)")
        
        # Single-pass if short
        if len(text) <= 3500:
            messages = [
                {"role": "system", "content": "You are an expert meeting summarizer. Produce a concise, faithful summary."},
                {"role": "user", "content": f"Summarize the following transcript in EXACTLY {target_words} words or less. Be specific, avoid fluff. The summary MUST be significantly shorter than the original transcript.\n\nTranscript:\n{text}"}
            ]
            resp = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=min(800, int(target_words * 2))  # Limit tokens based on target words
            )
            content = resp.choices[0].message.content.strip()
            # Strip code fences if any
            if content.startswith('```') and content.endswith('```'):
                content = content.split('\n',1)[1].rsplit('\n',1)[0].strip()
            return content
        
        # Multi-pass for long text
        chunk_size = 3500
        overlap = 300
        chunks: List[str] = []
        i = 0
        while i < len(text):
            chunks.append(text[i:i+chunk_size])
            i += chunk_size - overlap
        
        # Calculate per-chunk target words
        per_chunk_words = max(60, min(120, int(target_words / len(chunks)) + 20))
        
        chunk_summaries: List[str] = []
        for idx, ch in enumerate(chunks, 1):
            messages = [
                {"role": "system", "content": "You summarize meeting transcripts faithfully."},
                {"role": "user", "content": f"Summarize chunk {idx}/{len(chunks)} in {per_chunk_words} words or less, focusing on decisions, owners, and deadlines.\n\nChunk:\n{ch}"}
            ]
            resp = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=min(500, int(per_chunk_words * 2))
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith('```') and content.endswith('```'):
                content = content.split('\n',1)[1].rsplit('\n',1)[0].strip()
            chunk_summaries.append(content)
        
        synthesis_input = "\n\n".join(chunk_summaries)
        synth_messages = [
            {"role": "system", "content": "You are an expert meeting summarizer."},
            {"role": "user", "content": f"Given these chunk summaries, write a single, coherent summary of EXACTLY {target_words} words or less capturing objectives, decisions, owners, and deadlines. Avoid repetition. The final summary MUST be concise and significantly shorter than the original transcript.\n\nChunk summaries:\n{synthesis_input}"}
        ]
        final_resp = self.openai_client.chat.completions.create(
            model=model,
            messages=synth_messages,
            temperature=0,
            max_tokens=min(800, int(target_words * 2))
        )
        final_text = final_resp.choices[0].message.content.strip()
        if final_text.startswith('```') and final_text.endswith('```'):
            final_text = final_text.split('\n',1)[1].rsplit('\n',1)[0].strip()
        return final_text
    
    def extract_action_items(self, text: str, meeting_date: str = None, attendees: List[str] = None) -> Dict[str, Any]:
        """Extract action items, decisions, and key topics from text using OpenAI API or rule-based patterns as fallback.
        
        Returns a dictionary with:
        - action_items: List of actionable tasks with owners and deadlines
        - decisions: List of decisions made during the meeting
        - key_topics: List of main topics discussed
        - metadata: Extraction metadata (confidence, method used, etc.)
        """
        if not text:
            return {
                'action_items': [],
                'decisions': [],
                'key_topics': [],
                'metadata': {'method': 'none', 'confidence': 0.0, 'extraction_time': None}
            }
        
        # Try OpenAI extraction first
        if self.openai_client:
            try:
                result = self._extract_enhanced_items_openai(text, meeting_date, attendees)
                print(f"OpenAI enhanced extraction successful: {len(result.get('action_items', []))} actions, {len(result.get('decisions', []))} decisions, {len(result.get('key_topics', []))} topics")
                return result
            except Exception as e:
                import traceback
                print(f"OpenAI enhanced extraction failed: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                print("Falling back to rule-based extraction.")
        
        # Fallback to rule-based extraction
        return self._extract_enhanced_items_rule_based(text, meeting_date, attendees)
    
    def _extract_enhanced_items_openai(self, text: str, meeting_date: str = None, attendees: List[str] = None) -> Dict[str, Any]:
        """Extract action items, decisions, and key topics using OpenAI API with enhanced prompting."""
        from datetime import datetime
        start_time = datetime.now()
        
        meeting_date = meeting_date or datetime.now().isoformat()
        attendees_str = ", ".join(attendees) if attendees else "Unknown attendees"
        
        system_prompt = """You are an expert meeting assistant that extracts concrete, actionable tasks, decisions, and key topics from meeting transcripts.
You must return ONLY valid JSON matching the specified schema. Be precise and don't invent facts not present in the transcript."""
        
        def _normalize_for_model(t: str) -> str:
            if not t:
                return t
            t = unicodedata.normalize('NFKC', t)
            t = t.replace('\u2014', '-').replace('\u2013', '-')
            t = t.replace('\u2018', "'").replace('\u2019', "'")
            t = t.replace('\u201c', '"').replace('\u201d', '"')
            return t
            
        text_for_model = _normalize_for_model(text)
        
        user_prompt = f"""Extract action items, decisions, and key topics from this meeting transcript.

Meeting Context:
- Date/Time: {meeting_date}
- Attendees: {attendees_str}

Transcript:\n{text_for_model}\n
{text}

Instructions:
1. **ACTION ITEMS**: Extract concrete, actionable tasks with clear deliverables
   - Must have verb + object (what needs to be done)
   - Resolve pronouns to actual names when possible
   - Convert relative dates to ISO dates using meeting date as reference
   - Include confidence score (0-1) based on clarity
   - Priority: P1 (urgent/critical), P2 (important), P3 (routine)
   - Categories: UX, Infra, GTM, Ops, Research, Admin, Other

2. **DECISIONS**: Extract explicit decisions made during the meeting
   - Clear resolution or conclusion reached
   - Decision rationale when mentioned
   - Impact assessment if discussed
   - Who made the decision (if stated)

3. **KEY TOPICS**: Extract main discussion themes and subjects
   - Primary topics that consumed significant discussion time
   - Important themes or areas of focus
   - Strategic topics or areas of concern
   - Exclude trivial or very brief mentions

4. **EVIDENCE**: All items must include exact quotes from transcript
   - Quote must be verbatim from the source
   - If you cannot provide a verbatim quote, omit the item
   - Quotes should be 3-30 words for context

Return ONLY a JSON object matching this exact schema:
{{
  "action_items": [
    {{
      "text": "Clear, imperative description of the task",
      "owner": "Person name or null if unclear",
      "due_date_iso": "YYYY-MM-DD or null if not specified",
      "priority": "P1|P2|P3",
      "confidence": 0.95,
      "evidence_quote": "Exact quote from transcript",
      "char_start": 0,
      "char_end": 100,
      "category": "UX|Infra|GTM|Ops|Research|Admin|Other",
      "urgency_indicators": ["list of words/phrases indicating urgency"]
    }}
  ],
  "decisions": [
    {{
      "decision": "Clear statement of what was decided",
      "rationale": "Why this decision was made (if mentioned)",
      "decision_maker": "Person or group who made decision",
      "impact": "Expected impact or implications",
      "confidence": 0.90,
      "evidence_quote": "Exact quote supporting this decision",
      "char_start": 0,
      "char_end": 100,
      "category": "Strategic|Operational|Technical|Process|Other"
    }}
  ],
  "key_topics": [
    {{
      "topic": "Main topic or theme discussed",
      "description": "Brief summary of what was discussed",
      "duration_indicators": ["phrases suggesting extended discussion"],
      "importance_level": "High|Medium|Low",
      "confidence": 0.85,
      "evidence_quote": "Quote showing this topic was discussed",
      "char_start": 0,
      "char_end": 100,
      "category": "Strategic|Technical|Process|Business|Other"
    }}
  ],
  "metadata": {{
    "extraction_method": "openai",
    "model_used": "{os.getenv('OPENAI_ACTION_MODEL', self.openai_action_model)}",
    "total_confidence": 0.90,
    "processing_notes": ["Any relevant processing observations"]
  }}
}}

Rules:
- Focus on future actions, not past accomplishments
- Merge similar/duplicate items
- Exclude completed items ("already done", "finished", etc.)
- Limit to 20 most important items per category
- Every item MUST have a verbatim evidence quote
"""
        
        try:
            action_model = os.getenv('OPENAI_ACTION_MODEL', self.openai_action_model)
            print(f"Using OpenAI enhanced extraction model: {action_model}")
            
            response = self.openai_client.chat.completions.create(
                model=action_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=3000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                elif response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError:
                    first = response_text.find('{')
                    last = response_text.rfind('}')
                    if first != -1 and last != -1 and last > first:
                        result = json.loads(response_text[first:last+1])
                    else:
                        raise
            except json.JSONDecodeError as e:
                print(f"Failed to parse OpenAI response as JSON: {e}")
                print(f"Raw response: {response_text}")
                raise
            
            # Validate and enhance the response
            validated_result = self._validate_and_enhance_extraction(result, text, attendees)
            
            # Add processing metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            validated_result['metadata']['extraction_time'] = processing_time
            validated_result['metadata']['processing_timestamp'] = datetime.now().isoformat()
            
            return validated_result
            
        except Exception as e:
            print(f"OpenAI enhanced extraction failed: {e}")
            raise
            
    # Legacy method removed - using enhanced extraction instead
        
        except Exception as e:
            # If authentication failed, try to reinitialize client from .env and retry once
            msg = str(e)
            if '401' in msg or 'invalid_api_key' in msg.lower():
                try:
                    new_key = (dotenv_values(find_dotenv()) or {}).get('OPENAI_API_KEY')
                    if new_key and new_key != self.openai_api_key:
                        from openai import OpenAI
                        self.openai_api_key = new_key
                        self.openai_client = OpenAI(api_key=self.openai_api_key)
                        # Retry once
                        # Retry once with configured model
                        retry_model = os.getenv('OPENAI_ACTION_MODEL', self.openai_action_model)
                        print(f"Retrying OpenAI action extraction with model: {retry_model}")
                        response = self.openai_client.chat.completions.create(
                            model=retry_model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            max_tokens=2000,
                            temperature=0
                        )
                        response_text = response.choices[0].message.content.strip()
                        if response_text.startswith('```json'):
                            response_text = response_text[7:]
                        elif response_text.startswith('```'):
                            response_text = response_text[3:]
                        if response_text.endswith('```'):
                            response_text = response_text[:-3]
                        result = json.loads(response_text.strip())
                        action_items = result.get('action_items', [])
                        # Validate
                        validated_items = []
                        transcript = text or ""
                        transcript_lower = transcript.lower()
                        
                        def _build_flexible_regex(ev: str) -> Optional[re.Pattern]:
                            tokens = [t for t in re.findall(r"\w+", ev) if t]
                            if len(tokens) < 3 or len(tokens) > 25:
                                return None
                            parts = [r"\b" + re.escape(tok) + r"\b" for tok in tokens]
                            sep = r"[\s\.,;:!\?\-–—'\"/\\()\[\]]+"
                            pattern = sep.join(parts)
                            try:
                                return re.compile(pattern, re.IGNORECASE)
                            except Exception:
                                return None
                        
                        def find_span(ev: str):
                            if not ev:
                                return None
                            ev_norm = ev.strip()
                            idx = transcript_lower.find(ev_norm.lower())
                            if idx >= 0:
                                return idx, idx + len(ev_norm)
                            regex = _build_flexible_regex(ev_norm)
                            if regex:
                                m = regex.search(transcript)
                                if m:
                                    return m.start(), m.end()
                            return None
                        
                        attendee_set = set(attendees or [])
                        for item in action_items:
                            if isinstance(item, dict) and 'text' in item and item['text'].strip():
                                evidence = (item.get('evidence_quote') or '').strip()
                                span = find_span(evidence)
                                if not evidence or not span:
                                    continue
                                start, end = span
                                base_item = {
                                    'text': str(item.get('text', ''))[:300],
                                    'assignee': item.get('owner') or 'Unassigned',
                                    'deadline': item.get('due_date_iso') or 'No deadline specified',
                                    'priority': item.get('priority', 'P3'),
                                    'confidence': float(item.get('confidence', 0.5)),
                                    'evidence': evidence,
                                    'char_start': start,
                                    'char_end': end,
                                    'category': item.get('category', 'Other')
                                }
                                ev_text = evidence + ' ' + base_item['text']
                                if self._is_completed_statement(ev_text):
                                    continue
                                fragments = re.split(r";\s*", base_item['text'])
                                refined_fragments: List[str] = []
                                for frag in fragments:
                                    parts = re.split(r",?\s+and\s+(?=[A-Z][a-z]+\b)", frag)
                                    refined_fragments.extend([p.strip() for p in parts if p.strip()])
                                if len(refined_fragments) <= 1:
                                    refined_fragments = [base_item['text']]
                                for frag in refined_fragments:
                                    new_item = base_item.copy()
                                    new_item['text'] = frag[:300]
                                    for name in attendee_set:
                                        if re.search(rf"\b{re.escape(name)}\b", frag):
                                            new_item['assignee'] = name
                                            break
                                    new_item['confidence'] = max(0.0, min(1.0, new_item.get('confidence', 0.5) - 0.05))
                                    if new_item['confidence'] >= 0.5:
                                        validated_items.append(new_item)
                        validated_items.sort(key=lambda x: (x['priority'], -x['confidence']))
                        validated_items = self._merge_similar_items(validated_items)
                        return validated_items[:15]
                except Exception:
                    pass
            print(f"OpenAI API call failed: {e}")
            raise
    
    def _extract_action_items_rule_based(self, text: str) -> List[Dict[str, Any]]:
        """Extract action items from text using rule-based patterns (fallback method)."""
        if not text:
            return []
        
        action_items = []
        seen_items = set()  # To avoid duplicates
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
                
            # Check each pattern
            for pattern in self.action_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    # Extract the action text
                    action_text = ''
                    if len(match.groups()) >= 3:
                        action_text = match.group(3).strip()
                    elif len(match.groups()) >= 2:
                        action_text = match.group(2).strip()
                    elif len(match.groups()) >= 1:
                        action_text = match.group(1).strip()
                    
                    if not action_text:
                        continue
                    
                    # Clean up the action text
                    action_text = re.sub(r'^[-•*\s]+', '', action_text)
                    action_text = action_text.strip()
                    
                    # Split multi-owner/commitment fragments
                    fragments = re.split(r";\s*", action_text)
                    refined: List[str] = []
                    for frag in fragments:
                        parts = re.split(r",?\s+and\s+(?=[A-Z][a-z]+\b)", frag)
                        refined.extend([p.strip() for p in parts if p.strip()])
                    if not refined:
                        refined = [action_text]
                    
                    for frag in refined:
                        # Skip items that look already completed
                        if self._is_completed_statement(sentence) or self._is_completed_statement(frag):
                            continue
                        # Skip if too short or already seen
                        key = frag.lower()
                        if len(frag) > 10 and key not in seen_items:
                            # Try to extract assignee from the fragment first, then from full sentence
                            assignee = None
                            for token in (frag, sentence):
                                assignee = assignee or self._extract_assignee(token)
                                if assignee:
                                    break
                            # Try to extract deadline from the fragment
                            deadline = self._extract_deadline(frag) or self._extract_deadline(sentence)
                            action_item = {
                                'text': frag[:200],
                                'assignee': assignee,
                                'deadline': deadline
                            }
                            action_items.append(action_item)
                            seen_items.add(key)
                            if len(action_items) >= 10:
                                return action_items
        
        return action_items
    
    def _validate_and_enhance_extraction(self, result: Dict[str, Any], text: str, attendees: List[str] = None) -> Dict[str, Any]:
        """Validate and enhance the extracted items with evidence verification and quality checks."""
        transcript = text or ""
        transcript_lower = transcript.lower()
        attendee_set = set(attendees or [])
        
        def _build_flexible_regex(ev: str) -> Optional[re.Pattern]:
            tokens = [t for t in re.findall(r"\w+", ev) if t]
            if len(tokens) < 2 or len(tokens) > 25:
                return None
            parts = [r"\b" + re.escape(tok) + r"\b" for tok in tokens]
            sep = r"[\s\.,;:!\?\-–—'\"/\\()\[\]]+"
            pattern = sep.join(parts)
            try:
                return re.compile(pattern, re.IGNORECASE)
            except Exception:
                return None
        
        def find_span(ev: str):
            if not ev:
                return None
            ev_norm = ev.strip()
            # Simple case-insensitive exact match first
            idx = transcript_lower.find(ev_norm.lower())
            if idx >= 0:
                return idx, idx + len(ev_norm)
            # Flexible regex match
            regex = _build_flexible_regex(ev_norm)
            if regex:
                m = regex.search(transcript)
                if m:
                    return m.start(), m.end()
            return None
        
        validated_result = {
            'action_items': [],
            'decisions': [],
            'key_topics': [],
            'metadata': result.get('metadata', {'method': 'openai', 'confidence': 0.8})
        }
        
        # Validate action items
        for item in result.get('action_items', []):
            if isinstance(item, dict) and 'text' in item and item['text'].strip():
                evidence = (item.get('evidence_quote') or '').strip()
                span = find_span(evidence)
                if not evidence or not span:
                    continue
                
                start, end = span
                enhanced_item = {
                    'text': str(item.get('text', ''))[:300],
                    'owner': item.get('owner') or 'Unassigned',
                    'due_date_iso': item.get('due_date_iso') or None,
                    'priority': item.get('priority', 'P3'),
                    'confidence': float(item.get('confidence', 0.7)),
                    'evidence_quote': evidence,
                    'char_start': start,
                    'char_end': end,
                    'category': item.get('category', 'Other'),
                    'urgency_indicators': item.get('urgency_indicators', []),
                    'assignee': item.get('owner') or 'Unassigned',  # Legacy compatibility
                    'deadline': item.get('due_date_iso') or 'No deadline specified'  # Legacy compatibility
                }
                
                # Skip completed items
                if not self._is_completed_statement(evidence + ' ' + enhanced_item['text']):
                    validated_result['action_items'].append(enhanced_item)
        
        # Validate decisions
        for item in result.get('decisions', []):
            if isinstance(item, dict) and 'decision' in item and item['decision'].strip():
                evidence = (item.get('evidence_quote') or '').strip()
                span = find_span(evidence)
                if not evidence or not span:
                    continue
                    
                start, end = span
                enhanced_item = {
                    'decision': str(item.get('decision', ''))[:300],
                    'rationale': item.get('rationale') or 'Not specified',
                    'decision_maker': item.get('decision_maker') or 'Not specified',
                    'impact': item.get('impact') or 'Not specified',
                    'confidence': float(item.get('confidence', 0.7)),
                    'evidence_quote': evidence,
                    'char_start': start,
                    'char_end': end,
                    'category': item.get('category', 'Other')
                }
                validated_result['decisions'].append(enhanced_item)
        
        # Validate key topics
        for item in result.get('key_topics', []):
            if isinstance(item, dict) and 'topic' in item and item['topic'].strip():
                evidence = (item.get('evidence_quote') or '').strip()
                span = find_span(evidence)
                if not evidence or not span:
                    continue
                    
                start, end = span
                enhanced_item = {
                    'topic': str(item.get('topic', ''))[:200],
                    'description': item.get('description') or 'No description provided',
                    'duration_indicators': item.get('duration_indicators', []),
                    'importance_level': item.get('importance_level', 'Medium'),
                    'confidence': float(item.get('confidence', 0.6)),
                    'evidence_quote': evidence,
                    'char_start': start,
                    'char_end': end,
                    'category': item.get('category', 'Other')
                }
                validated_result['key_topics'].append(enhanced_item)
        
        # Sort by confidence and priority
        validated_result['action_items'].sort(key=lambda x: (x['priority'], -x['confidence']))
        validated_result['decisions'].sort(key=lambda x: -x['confidence'])
        validated_result['key_topics'].sort(key=lambda x: (-x['confidence'], x['importance_level'] == 'High'))
        
        # Limit results
        validated_result['action_items'] = validated_result['action_items'][:20]
        validated_result['decisions'] = validated_result['decisions'][:15]
        validated_result['key_topics'] = validated_result['key_topics'][:10]
        
        return validated_result
    
    def _extract_enhanced_items_rule_based(self, text: str, meeting_date: str = None, attendees: List[str] = None) -> Dict[str, Any]:
        """Enhanced rule-based extraction for action items, decisions, and topics as fallback."""
        if not text:
            return {
                'action_items': [],
                'decisions': [],
                'key_topics': [],
                'metadata': {'method': 'rule_based', 'confidence': 0.5, 'extraction_time': None}
            }
        
        from datetime import datetime
        start_time = datetime.now()
        
        # Use legacy rule-based action item extraction
        legacy_action_items = self._extract_action_items_rule_based(text)
        
        # Extract decisions using rule-based patterns
        decision_patterns = [
            r'\b(decided|agreed|concluded|determined|resolved)\s+(?:to|that|on)\s+([^.!?]{10,100})',
            r'\b(decision|resolution|agreement)[:]*\s*([^.!?]+)',
            r'\b(we\s+will|let\'s|going\s+to|plan\s+to)\s+([^.!?]{10,100})',
            r'\b(consensus|unanimous|majority)\s+(?:is|was|to)\s+([^.!?]{10,100})'
        ]
        
        decisions = []
        for pattern in decision_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                decision_text = match.group(2).strip() if len(match.groups()) >= 2 else match.group(1).strip()
                if len(decision_text) > 10:
                    decisions.append({
                        'decision': decision_text[:300],
                        'rationale': 'Not specified',
                        'decision_maker': 'Not specified',
                        'impact': 'Not specified',
                        'confidence': 0.6,
                        'evidence_quote': match.group(0)[:100],
                        'char_start': match.start(),
                        'char_end': match.end(),
                        'category': 'Other'
                    })
        
        # Extract key topics using rule-based patterns
        topic_patterns = [
            r'\b(discuss|talking\s+about|focus\s+on|regarding|concerning)\s+([^.!?]{5,50})',
            r'\b(topic|subject|issue|matter|question)\s*[:]*\s*([^.!?]+)',
            r'\b(main\s+point|key\s+issue|important\s+aspect)\s*[:]*\s*([^.!?]+)'
        ]
        
        topics = []
        topic_set = set()
        for pattern in topic_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                topic_text = match.group(2).strip() if len(match.groups()) >= 2 else match.group(1).strip()
                topic_key = topic_text.lower()[:50]
                if len(topic_text) > 5 and topic_key not in topic_set:
                    topic_set.add(topic_key)
                    topics.append({
                        'topic': topic_text[:200],
                        'description': 'Identified from meeting discussion',
                        'duration_indicators': [],
                        'importance_level': 'Medium',
                        'confidence': 0.5,
                        'evidence_quote': match.group(0)[:100],
                        'char_start': match.start(),
                        'char_end': match.end(),
                        'category': 'Other'
                    })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'action_items': legacy_action_items,
            'decisions': decisions[:15],
            'key_topics': topics[:10],
            'metadata': {
                'method': 'rule_based',
                'confidence': 0.5,
                'extraction_time': processing_time,
                'processing_timestamp': datetime.now().isoformat()
            }
        }
    
    def _is_completed_statement(self, text: str) -> bool:
        """Heuristic to detect statements that indicate the work is already completed."""
        if not text:
            return False
        t = text.lower()
        completed_markers = [
            "already", "has been", "have been", "was completed", "were completed",
            "finalized", "finished", "done", "ready to be shared", "is ready", "are ready",
            "created yesterday", "yesterday", "we have", "we've",
        ]
        # Avoid false positives when paired with future modals
        future_markers = ["will ", "need to", "must ", "should ", "by "]
        if any(m in t for m in completed_markers) and not any(f in t for f in future_markers):
            return True
        return False
    
    def _merge_similar_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge similar action items to avoid duplicates."""
        if len(items) <= 1:
            return items
        
        merged = []
        processed_indices = set()
        
        for i, item1 in enumerate(items):
            if i in processed_indices:
                continue
                
            # Start with the current item
            merged_item = item1.copy()
            similar_items = [item1]
            
            # Find similar items
            for j, item2 in enumerate(items[i+1:], start=i+1):
                if j in processed_indices:
                    continue
                    
                # Calculate similarity
                similarity = difflib.SequenceMatcher(None, 
                    item1['text'].lower(), 
                    item2['text'].lower()).ratio()
                
                # If similar enough, merge
                if similarity > 0.7:
                    similar_items.append(item2)
                    processed_indices.add(j)
                    
                    # Keep the better fields
                    if item2.get('assignee') and item2['assignee'] != 'Unassigned':
                        merged_item['assignee'] = item2['assignee']
                    if item2.get('deadline') and item2['deadline'] != 'No deadline specified':
                        merged_item['deadline'] = item2['deadline']
                    if item2.get('confidence', 0) > merged_item.get('confidence', 0):
                        merged_item['confidence'] = item2['confidence']
                        merged_item['evidence'] = item2.get('evidence', '')
            
            # Average confidence across similar items
            if len(similar_items) > 1 and 'confidence' in merged_item:
                total_conf = sum(item.get('confidence', 0.5) for item in similar_items)
                merged_item['confidence'] = total_conf / len(similar_items)
            
            merged.append(merged_item)
            processed_indices.add(i)
        
        return merged
    
    def _normalize_dates(self, text: str, reference_date: datetime = None) -> str:
        """Convert relative dates to absolute dates."""
        if not reference_date:
            reference_date = datetime.now()
        
        # Map of relative terms to days
        relative_mappings = {
            'today': 0,
            'tomorrow': 1,
            'day after tomorrow': 2,
            'next monday': None,  # Special handling
            'next tuesday': None,
            'next wednesday': None,
            'next thursday': None,
            'next friday': None,
            'next week': 7,
            'in two weeks': 14,
            'next month': 30,
            'end of week': None,  # Special handling
            'eod': 0,  # End of day today
            'eow': None,  # End of week
            'eom': None,  # End of month
        }
        
        normalized_text = text.lower()
        
        # Handle "next [day]" patterns
        for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
            if f'next {day}' in normalized_text:
                # Find the next occurrence of this day
                days_ahead = 0
                current_day = reference_date.weekday()
                target_day = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'].index(day)
                
                if target_day <= current_day:
                    days_ahead = 7 - current_day + target_day
                else:
                    days_ahead = target_day - current_day
                
                target_date = reference_date + timedelta(days=days_ahead)
                return target_date.strftime('%Y-%m-%d')
        
        # Handle other relative dates
        for term, days in relative_mappings.items():
            if term in normalized_text and days is not None:
                target_date = reference_date + timedelta(days=days)
                return target_date.strftime('%Y-%m-%d')
        
        # If no conversion possible, return original
        return text
    
    def _extract_assignee(self, text: str) -> str:
        """Extract assignee from text."""
        # Look for names (capitalized words) before action verbs
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:will|should|needs to|must|to)\s+'
        match = re.search(pattern, text)
        if match:
            name = match.group(1)
            # Filter out common non-name words
            if name.lower() not in ['the', 'we', 'they', 'it', 'this', 'that']:
                return name
        return None
    
    def _extract_deadline(self, text: str) -> str:
        """Extract deadline from text."""
        # Look for date patterns
        patterns = [
            r'by\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)',
            r'by\s+(tomorrow|today|next week|next month|end of day|EOD|COB)',
            r'by\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d+',
            r'by\s+(\d{1,2}/\d{1,2}(?:/\d{2,4})?)',
            r'before\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)',
            r'deadline[:\s]+([\w\s]+?)(?:[,.]|$)',
            r'due\s+([\w\s]+?)(?:[,.]|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract keywords using TF-IDF."""
        if not text:
            return []
        
        # Tokenize and clean text
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter out stop words and short words
        words = [w for w in words if w not in self.stop_words and len(w) > 3]
        
        if not words:
            return []
        
        # Calculate word frequencies
        word_freq = Counter(words)
        total_words = len(words)
        
        # Calculate TF-IDF scores (simplified version)
        tfidf_scores = {}
        for word, freq in word_freq.items():
            tf = freq / total_words
            # Simple IDF: penalize very common words
            idf = math.log(total_words / (1 + freq)) if freq < total_words * 0.5 else 0.1
            tfidf_scores[word] = tf * idf
        
        # Get top keywords
        top_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N keywords
        return [word for word, score in top_keywords[:num_keywords]]
