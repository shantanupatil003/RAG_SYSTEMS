# context_only_rag.py - Strictly Context-Based Biology RAG System
import streamlit as st
import ollama
import PyPDF2
import os
import json
import re
from typing import List, Dict, Tuple
from datetime import datetime

class ContextStrictDocumentProcessor:
    """Processes documents with strict context-only focus"""
    
    def __init__(self, data_directory: str = "data"):
        self.data_directory = data_directory
        self.knowledge_base = []
        self.metadata = {}
        self.knowledge_file = "context_only_knowledge.json"
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF files"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} of {pdf_path} ---\n{page_text}\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from TXT files"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                content = file.read()
                return f"\n--- Content from {txt_path} ---\n{content}\n"
        except Exception as e:
            st.error(f"Error reading TXT {txt_path}: {e}")
            return ""
    
    def create_context_chunks(self, text: str, filename: str) -> List[Dict]:
        """Create chunks that preserve context and source information"""
        chunks = []
        
        # Clean the text but preserve structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Remove excessive newlines
        
        # Split by sections while preserving source info
        section_patterns = [
            r'--- Page \d+ of [^-]+ ---',  # Page breaks
            r'\n(?:Chapter|Section|Part)\s+\d+[^\n]*\n',
            r'\n\d+\.\d+\s+[A-Z][^\n]*\n',  # Numbered sections
            r'\n[A-Z][A-Z\s]{10,}[A-Z]\n',  # ALL CAPS headers
        ]
        
        sections = [text]
        for pattern in section_patterns:
            new_sections = []
            for section in sections:
                parts = re.split(f'({pattern})', section, flags=re.IGNORECASE)
                # Keep the separators with the following content
                for i in range(0, len(parts), 2):
                    if i + 1 < len(parts):
                        combined = parts[i] + parts[i+1]
                        if combined.strip():
                            new_sections.append(combined)
                    else:
                        if parts[i].strip():
                            new_sections.append(parts[i])
            sections = new_sections
        
        # Create chunks with maximum context preservation
        for section in sections:
            section = section.strip()
            if len(section) < 100:  # Skip very short sections
                continue
                
            if len(section) <= 1800:  # Good size for context preservation
                chunks.append({
                    'content': section,
                    'source': filename,
                    'chunk_id': len(chunks),
                    'char_count': len(section),
                    'source_location': self._extract_source_location(section, filename)
                })
            else:
                # Split longer sections but keep context
                paragraphs = section.split('\n\n')
                current_chunk = ""
                source_info = self._extract_source_location(section[:200], filename)
                
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    
                    if len(current_chunk + '\n\n' + para) <= 1800:
                        current_chunk += '\n\n' + para if current_chunk else para
                    else:
                        if current_chunk and len(current_chunk) > 150:
                            chunks.append({
                                'content': current_chunk,
                                'source': filename,
                                'chunk_id': len(chunks),
                                'char_count': len(current_chunk),
                                'source_location': source_info
                            })
                        current_chunk = para
                        source_info = self._extract_source_location(para[:100], filename)
                
                if current_chunk and len(current_chunk) > 150:
                    chunks.append({
                        'content': current_chunk,
                        'source': filename,
                        'chunk_id': len(chunks),
                        'char_count': len(current_chunk),
                        'source_location': source_info
                    })
        
        return chunks
    
    def _extract_source_location(self, text: str, filename: str) -> str:
        """Extract specific source location from text"""
        # Look for page references
        page_match = re.search(r'--- Page (\d+) of [^-]+ ---', text)
        if page_match:
            return f"{filename}, Page {page_match.group(1)}"
        
        # Look for chapter/section references
        chapter_match = re.search(r'(?:Chapter|Section)\s+(\d+)', text, re.IGNORECASE)
        if chapter_match:
            return f"{filename}, Chapter/Section {chapter_match.group(1)}"
        
        return filename
    
    def process_all_documents(self):
        """Process all documents with strict context tracking"""
        if not os.path.exists(self.data_directory):
            st.error(f"Data directory '{self.data_directory}' not found!")
            return False
        
        self.knowledge_base = []
        files_processed = 0
        
        st.info(f"📚 Processing documents for context-only system...")
        
        files = [f for f in os.listdir(self.data_directory) 
                if f.lower().endswith(('.pdf', '.txt')) and os.path.isfile(os.path.join(self.data_directory, f))]
        
        if not files:
            st.warning("No PDF or TXT files found in data directory!")
            return False
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, filename in enumerate(files):
            file_path = os.path.join(self.data_directory, filename)
            status_text.text(f"Processing: {filename}")
            
            try:
                # Extract text with source tracking
                if filename.lower().endswith('.pdf'):
                    text = self.extract_text_from_pdf(file_path)
                elif filename.lower().endswith('.txt'):
                    text = self.extract_text_from_txt(file_path)
                else:
                    continue
                
                if not text.strip():
                    st.warning(f"No text extracted from {filename}")
                    continue
                
                # Create context-preserving chunks
                chunks = self.create_context_chunks(text, filename)
                
                if chunks:
                    self.knowledge_base.extend(chunks)
                    files_processed += 1
                    
                    # Store detailed metadata
                    self.metadata[filename] = {
                        'chunks_count': len(chunks),
                        'processed_at': datetime.now().isoformat(),
                        'total_characters': sum(chunk['char_count'] for chunk in chunks),
                        'file_type': filename.split('.')[-1].lower(),
                        'avg_chunk_size': sum(chunk['char_count'] for chunk in chunks) // len(chunks),
                        'source_locations': list(set(chunk['source_location'] for chunk in chunks))
                    }
                    
                    st.success(f"✅ {filename}: {len(chunks)} context chunks created")
                else:
                    st.warning(f"No chunks created from {filename}")
            
            except Exception as e:
                st.error(f"❌ Error processing {filename}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(files))
        
        status_text.text("✅ Context processing complete!")
        self.save_knowledge_base()
        
        if files_processed > 0:
            st.success(f"🎉 Successfully processed {files_processed} files with {len(self.knowledge_base)} context chunks!")
            return True
        return False
    
    def save_knowledge_base(self):
        """Save the context-only knowledge base"""
        try:
            data = {
                'knowledge_base': self.knowledge_base,
                'metadata': self.metadata,
                'created_at': datetime.now().isoformat(),
                'version': '1.0',
                'mode': 'context_only'
            }
            
            with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"Error saving knowledge base: {e}")
    
    def load_knowledge_base(self) -> bool:
        """Load existing context-only knowledge base"""
        try:
            if os.path.exists(self.knowledge_file):
                with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('mode') == 'context_only':
                        self.knowledge_base = data.get('knowledge_base', [])
                        self.metadata = data.get('metadata', {})
                        return len(self.knowledge_base) > 0
        except Exception as e:
            st.error(f"Error loading knowledge base: {e}")
        return False

class ContextOnlyRetriever:
    """Retriever that focuses purely on context matching"""
    
    def __init__(self, knowledge_base: List[Dict]):
        self.knowledge_base = knowledge_base
    
    def calculate_context_relevance(self, query: str, chunk: Dict) -> Tuple[float, str]:
        """Calculate relevance based only on context matching"""
        query_lower = query.lower()
        content_lower = chunk['content'].lower()
        
        score = 0.0
        match_details = []
        
        # Exact phrase matching (highest priority)
        if query_lower in content_lower:
            score += 15.0
            match_details.append("exact phrase match")
        
        # Word matching with context
        query_words = [w for w in re.findall(r'\b\w+\b', query_lower) if len(w) > 2]
        content_words = set(re.findall(r'\b\w+\b', content_lower))
        
        if query_words:
            matches = [w for w in query_words if w in content_words]
            if matches:
                match_ratio = len(matches) / len(query_words)
                score += match_ratio * 10.0
                match_details.append(f"word matches: {', '.join(matches[:5])}")
        
        # Boost for longer, more comprehensive chunks
        if len(chunk['content']) > 800:
            score *= 1.2
            match_details.append("comprehensive content")
        
        # Context quality bonus
        if chunk.get('source_location'):
            score += 1.0
            match_details.append(f"from {chunk['source_location']}")
        
        match_explanation = "; ".join(match_details) if match_details else "no clear matches"
        
        return score, match_explanation
    
    def retrieve_context_chunks(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float, str]]:
        """Retrieve chunks with context matching details"""
        if not self.knowledge_base:
            return []
        
        scored_chunks = []
        for chunk in self.knowledge_base:
            score, explanation = self.calculate_context_relevance(query, chunk)
            if score > 0:
                scored_chunks.append((chunk, score, explanation))
        
        # Sort by score and return top_k with explanations
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

class ContextOnlyRAGSystem:
    """RAG system that ONLY uses document context"""
    
    def __init__(self):
        self.processor = ContextStrictDocumentProcessor()
        self.retriever = None
        self.model_name = "llama2"
    
    def initialize_system(self) -> bool:
        """Initialize the context-only system"""
        if self.processor.load_knowledge_base():
            st.success("📚 Loaded existing context-only knowledge base!")
            self.retriever = ContextOnlyRetriever(self.processor.knowledge_base)
            return True
        else:
            if self.processor.process_all_documents():
                self.retriever = ContextOnlyRetriever(self.processor.knowledge_base)
                return True
            return False
    
    def get_available_models(self) -> List[str]:
        """Get available Ollama models"""
        try:
            result = ollama.list()
            models = [model['name'] for model in result.get('models', [])]
            return models if models else ['llama2']
        except:
            return ['llama2', 'mistral', 'codellama']
    
    def create_strict_context_prompt(self, query: str, context_chunks: List[Tuple[Dict, float, str]]) -> str:
        """Create a prompt that ONLY allows context-based answers"""
        
        if not context_chunks:
            return f"""CRITICAL INSTRUCTION: You are a document-based assistant that can ONLY answer questions using information explicitly provided in the document context.

USER QUESTION: "{query}"

AVAILABLE CONTEXT: None

RESPONSE INSTRUCTION: Since no relevant context was found in the provided documents, you MUST respond with:
"I cannot answer this question because I did not find any relevant information about '{query}' in the provided documents. I can only answer questions based on the content of the uploaded PDF and text files."

Do NOT use any external knowledge or general information. Only respond based on what is explicitly in the document context."""
        
        # Build strict context sections
        context_sections = []
        sources_info = []
        
        for i, (chunk, score, explanation) in enumerate(context_chunks, 1):
            sources_info.append(f"Source {i}: {chunk['source_location']} (relevance: {score:.1f}, {explanation})")
            context_sections.append(f"""
CONTEXT SECTION {i}:
Source: {chunk['source_location']}
Content: {chunk['content']}
""")
        
        context_text = "\n".join(context_sections)
        sources_list = "\n".join(sources_info)
        
        prompt = f"""CRITICAL INSTRUCTIONS: You are a document-based assistant with STRICT RULES:

1. You can ONLY use information explicitly stated in the CONTEXT SECTIONS below
2. You MUST NOT use any external knowledge, general information, or assumptions
3. If the context doesn't contain enough information to fully answer the question, you MUST say so clearly
4. You MUST cite which specific context section(s) you're using
5. If you're uncertain about any information, state your uncertainty
6. Do NOT make up, infer, or assume any information not explicitly in the context

CONTEXT SECTIONS FROM YOUR DOCUMENTS:
{context_text}

SOURCE INFORMATION:
{sources_list}

USER QUESTION: {query}

RESPONSE REQUIREMENTS:
- Answer ONLY based on the context sections above
- Cite which context section(s) you're using (e.g., "According to Context Section 1...")
- If context is insufficient, state: "The provided documents contain some relevant information, but not enough to fully answer this question."
- Be precise and stick to what's explicitly stated in the documents
- Do NOT add external knowledge or general information

Your response:"""
        
        return prompt
    
    def generate_context_only_response(self, query: str) -> Dict:
        """Generate response using ONLY document context"""
        
        if not self.retriever:
            return {
                'answer': "❌ System not initialized. Please process documents first.",
                'sources': [],
                'confidence': 0.0,
                'context_used': False,
                'error': True
            }
        
        try:
            # Retrieve context chunks
            context_chunks = self.retriever.retrieve_context_chunks(query, top_k=5)
            
            # Create strict context-only prompt
            prompt = self.create_strict_context_prompt(query, context_chunks)
            
            try:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'temperature': 0.3,  # Lower temperature for more precise context adherence
                        'top_p': 0.8,
                        'num_predict': 1000
                    }
                )
                
                answer = response['response'].strip()
                
                # Calculate context confidence
                if context_chunks:
                    avg_score = sum(score for _, score, _ in context_chunks) / len(context_chunks)
                    confidence = min(avg_score / 15.0, 1.0)
                    context_used = True
                else:
                    confidence = 0.0
                    context_used = False
                
                return {
                    'answer': answer,
                    'sources': [chunk['content'] for chunk, _, _ in context_chunks],
                    'source_locations': [chunk['source_location'] for chunk, _, _ in context_chunks],
                    'match_explanations': [explanation for _, _, explanation in context_chunks],
                    'confidence': confidence,
                    'context_used': context_used,
                    'context_chunks_found': len(context_chunks),
                    'error': False
                }
                
            except Exception as e:
                return {
                    'answer': f"❌ Error generating response: {str(e)}\n\nTry: ollama pull {self.model_name}",
                    'sources': [],
                    'confidence': 0.0,
                    'context_used': False,
                    'error': True
                }
                
        except Exception as e:
            return {
                'answer': f"❌ System error: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'context_used': False,
                'error': True
            }

def main():
    """Main application for context-only RAG system"""
    
    st.set_page_config(
        page_title="Context-Only RAG System",
        page_icon="📖",
        layout="wide"
    )
    
    st.title("📖 Context-Only RAG System")
    st.markdown("*Answers ONLY from your uploaded documents - No external knowledge*")
    
    # Warning about context-only mode
    st.warning("🔒 **Context-Only Mode**: This system will ONLY answer questions using information explicitly found in your PDF/text files. It will NOT use any external knowledge or general information.")
    
    # Initialize session state
    if 'context_rag_system' not in st.session_state:
        st.session_state.context_rag_system = ContextOnlyRAGSystem()
        st.session_state.system_initialized = False
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.header("📚 Context-Only Control")
        
        # System initialization
        if not st.session_state.system_initialized:
            st.warning("⚠️ Context system not initialized")
            if st.button("🚀 Initialize Context-Only System", type="primary"):
                with st.spinner("Processing documents for context-only mode..."):
                    if st.session_state.context_rag_system.initialize_system():
                        st.session_state.system_initialized = True
                        st.success("✅ Context-only system ready!")
                        st.rerun()
                    else:
                        st.error("Failed to initialize system")
        else:
            st.success("✅ Context-Only System Ready")
            
            if st.button("🔄 Reprocess Documents"):
                st.session_state.context_rag_system.processor.knowledge_base = []
                if st.session_state.context_rag_system.processor.process_all_documents():
                    st.session_state.context_rag_system.retriever = ContextOnlyRetriever(
                        st.session_state.context_rag_system.processor.knowledge_base
                    )
                    st.success("Documents reprocessed!")
                    st.rerun()
        
        st.markdown("---")
        
        # Model selection
        st.subheader("🤖 Model Settings")
        available_models = st.session_state.context_rag_system.get_available_models()
        selected_model = st.selectbox(
            "Select LLM Model",
            available_models,
            index=available_models.index(st.session_state.context_rag_system.model_name) 
            if st.session_state.context_rag_system.model_name in available_models else 0
        )
        
        if selected_model != st.session_state.context_rag_system.model_name:
            st.session_state.context_rag_system.model_name = selected_model
            st.success(f"Model changed to {selected_model}")
        
        st.markdown("---")
        
        # Knowledge base info
        st.subheader("📊 Document Context")
        if st.session_state.system_initialized:
            kb_size = len(st.session_state.context_rag_system.processor.knowledge_base)
            st.metric("Total Context Chunks", kb_size)
            
            if st.session_state.context_rag_system.processor.metadata:
                with st.expander("📋 Document Sources"):
                    for filename, info in st.session_state.context_rag_system.processor.metadata.items():
                        st.write(f"**{filename}**")
                        st.write(f"- Context chunks: {info['chunks_count']}")
                        st.write(f"- Total characters: {info['total_characters']:,}")
                        st.write(f"- Avg chunk size: {info['avg_chunk_size']} chars")
                        if info.get('source_locations'):
                            st.write(f"- Locations: {len(info['source_locations'])} sections")
        else:
            st.metric("Total Context Chunks", "Not initialized")
        
        # Context-only reminder
        st.markdown("---")
        st.info("🔒 **Remember**: This system only uses information from your uploaded documents. It cannot access external knowledge or general information.")
    
    # Main interface
    if st.session_state.system_initialized:
        st.header("💬 Ask Questions About Your Documents")
        
        # Display conversation
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant" and "sources" in message:
                    # Show context information
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if message.get("context_used"):
                            st.success(f"✅ Context Used")
                        else:
                            st.warning("❌ No Context Found")
                    with col2:
                        chunks_found = message.get("context_chunks_found", 0)
                        st.info(f"📚 {chunks_found} relevant chunks")
                    with col3:
                        confidence = message.get("confidence", 0)
                        color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
                        st.metric("Confidence", f"{color} {confidence:.2f}")
                    
                    if message.get("sources"):
                        with st.expander("📖 View Document Context Used"):
                            for i, (source, location, explanation) in enumerate(zip(
                                message["sources"][:3],
                                message.get("source_locations", []),
                                message.get("match_explanations", [])
                            ), 1):
                                st.markdown(f"**Context {i} (from {location}):**")
                                st.markdown(f"*Match: {explanation}*")
                                st.text_area(
                                    f"Document Content {i}",
                                    source[:600] + "..." if len(source) > 600 else source,
                                    height=200,
                                    key=f"context_{message.get('timestamp', '')}_{i}"
                                )
        
        # Chat input
        if prompt := st.chat_input("Ask about information in your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("📖 Searching document context..."):
                    result = st.session_state.context_rag_system.generate_context_only_response(prompt)
                
                st.markdown(result["answer"])
                
                # Add to conversation with all context info
                message_data = {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "source_locations": result.get("source_locations", []),
                    "match_explanations": result.get("match_explanations", []),
                    "confidence": result.get("confidence", 0),
                    "context_used": result.get("context_used", False),
                    "context_chunks_found": result.get("context_chunks_found", 0),
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(message_data)
        
        # Example questions
        st.markdown("---")
        st.markdown("### 💡 Try These Document-Based Questions:")
        
        example_questions = [
            "What information about the heart is in these documents?",
            "What does the document say about heart structure?",
            "According to the documents, how does circulation work?",
            "What biology concepts are covered in these files?",
            "What specific details about heart anatomy are mentioned?",
            "What does the document explain about cardiovascular function?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(question, key=f"context_example_{i}"):
                    st.session_state.messages.append({"role": "user", "content": question})
                    with st.spinner("📖 Searching document context..."):
                        result = st.session_state.context_rag_system.generate_context_only_response(question)
                    
                    message_data = {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                        "source_locations": result.get("source_locations", []),
                        "match_explanations": result.get("match_explanations", []),
                        "confidence": result.get("confidence", 0),
                        "context_used": result.get("context_used", False),
                        "context_chunks_found": result.get("context_chunks_found", 0),
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.messages.append(message_data)
                    st.rerun()
    
    else:
        st.info("🚀 Click 'Initialize Context-Only System' to get started!")
        
        st.markdown("""
        ### 📖 Context-Only Features:
        - **Strict Document-Only Answers**: No external knowledge used
        - **Source Transparency**: Shows exactly which document sections were used
        - **Context Matching**: Explains why each source was selected
        - **Confidence Scoring**: Based on context relevance only
        - **Clear Limitations**: Will clearly state when documents don't contain enough information
        
        ### 🔒 What This Means:
        - Answers come ONLY from your PDF/text files
        - No general knowledge or assumptions
        - Clear attribution to specific document sections
        - Will refuse to answer if context is insufficient
        """)

if __name__ == "__main__":
    main()