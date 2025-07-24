# biology_rag.py - Biology-Focused RAG-LLM System
import streamlit as st
import ollama
import PyPDF2
import os
import json
import re
from typing import List, Dict, Tuple
from datetime import datetime

class BiologyDocumentProcessor:
    """Processes biology and medical documents for RAG system"""
    
    def __init__(self, data_directory: str = "data"):
        self.data_directory = data_directory
        self.knowledge_base = []
        self.metadata = {}
        self.knowledge_file = "biology_knowledge_base.json"
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF files"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from TXT files"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            st.error(f"Error reading TXT {txt_path}: {e}")
            return ""
    
    def intelligent_chunk_biology_content(self, text: str, filename: str) -> List[Dict]:
        """Create intelligent chunks optimized for biology/medical content"""
        chunks = []
        
        # Clean the text
        text = re.sub(r'\n+', '\n', text)  # Remove excessive newlines
        text = re.sub(r'\s+', ' ', text)   # Normalize whitespace
        
        # Biology/medical specific chunking patterns
        section_patterns = [
            r'\n(?:Chapter|Section|Part)\s+\d+',
            r'\n\d+\.\d+\s+[A-Z]',  # 1.1 Section headers
            r'\n[A-Z][^.!?]*(?:Introduction|Overview|Summary|Conclusion|Definition|Function|Structure|Process|System|Anatomy|Physiology)',
            r'\n[A-Z]{2,}[^a-z]*\n',  # ALL CAPS headers
            r'\nFigure\s+\d+',   # Figures
            r'\nTable\s+\d+',    # Tables
            r'\n(?:The\s+)?(?:Heart|Cardiac|Cardiovascular|Blood|Circulation|Ventricle|Atrium|Artery|Vein)',  # Heart-specific sections
        ]
        
        # Split by biology section patterns
        sections = [text]  # Start with full text
        for pattern in section_patterns:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section)
                new_sections.extend(parts)
            sections = [s.strip() for s in new_sections if s.strip() and len(s) > 100]
        
        # Further process sections to create optimal chunks
        for section in sections:
            if len(section) <= 1500:  # Good size for biology content
                chunks.append({
                    'content': section,
                    'source': filename,
                    'chunk_id': len(chunks),
                    'type': 'biology_section',
                    'topics': self._extract_biology_topics(section)
                })
            else:
                # Split long sections by paragraphs, keeping related content together
                paragraphs = section.split('\n\n')
                current_chunk = ""
                
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    
                    # Check if adding this paragraph keeps chunk at good size
                    if len(current_chunk + para) < 1400:
                        current_chunk += para + "\n\n"
                    else:
                        # Save current chunk if it has substantial content
                        if len(current_chunk.strip()) > 200:
                            chunks.append({
                                'content': current_chunk.strip(),
                                'source': filename,
                                'chunk_id': len(chunks),
                                'type': 'biology_section',
                                'topics': self._extract_biology_topics(current_chunk)
                            })
                        current_chunk = para + "\n\n"
                
                # Add the last chunk
                if len(current_chunk.strip()) > 200:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'source': filename,
                        'chunk_id': len(chunks),
                        'type': 'biology_section',
                        'topics': self._extract_biology_topics(current_chunk)
                    })
        
        return chunks
    
    def _extract_biology_topics(self, text: str) -> List[str]:
        """Extract biology-related topics from text"""
        biology_keywords = [
            'heart', 'cardiac', 'cardiovascular', 'blood', 'circulation', 'ventricle', 'atrium',
            'artery', 'vein', 'capillary', 'pulse', 'rhythm', 'beat', 'pressure', 'flow',
            'anatomy', 'physiology', 'structure', 'function', 'system', 'organ', 'tissue',
            'cell', 'muscle', 'nerve', 'vessel', 'chamber', 'valve', 'membrane', 'wall',
            'oxygen', 'carbon dioxide', 'hemoglobin', 'plasma', 'red blood cell', 'white blood cell',
            'disease', 'condition', 'disorder', 'syndrome', 'treatment', 'therapy', 'diagnosis'
        ]
        
        text_lower = text.lower()
        found_topics = []
        
        for keyword in biology_keywords:
            if keyword in text_lower:
                found_topics.append(keyword)
        
        return found_topics
    
    def process_all_documents(self):
        """Process all biology documents in the data directory"""
        if not os.path.exists(self.data_directory):
            st.error(f"Data directory '{self.data_directory}' not found!")
            return
        
        self.knowledge_base = []
        files_processed = 0
        
        progress_container = st.container()
        
        with progress_container:
            st.info(f"🔬 Processing biology documents from '{self.data_directory}' directory...")
            
            files = [f for f in os.listdir(self.data_directory) 
                    if f.lower().endswith(('.pdf', '.txt')) and os.path.isfile(os.path.join(self.data_directory, f))]
            
            if not files:
                st.warning("No PDF or TXT files found in data directory!")
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, filename in enumerate(files):
                file_path = os.path.join(self.data_directory, filename)
                status_text.text(f"Processing: {filename}")
                
                try:
                    # Extract text based on file type
                    if filename.lower().endswith('.pdf'):
                        text = self.extract_text_from_pdf(file_path)
                    elif filename.lower().endswith('.txt'):
                        text = self.extract_text_from_txt(file_path)
                    else:
                        continue
                    
                    if not text.strip():
                        st.warning(f"No text extracted from {filename}")
                        continue
                    
                    # Create intelligent chunks for biology content
                    chunks = self.intelligent_chunk_biology_content(text, filename)
                    
                    if chunks:
                        self.knowledge_base.extend(chunks)
                        files_processed += 1
                        
                        # Store metadata
                        self.metadata[filename] = {
                            'chunks_count': len(chunks),
                            'processed_at': datetime.now().isoformat(),
                            'file_size': len(text),
                            'file_type': filename.split('.')[-1].lower(),
                            'topics_found': list(set([topic for chunk in chunks for topic in chunk['topics']]))
                        }
                        
                        st.success(f"✅ {filename}: {len(chunks)} chunks created")
                    else:
                        st.warning(f"No chunks created from {filename}")
                
                except Exception as e:
                    st.error(f"❌ Error processing {filename}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(files))
            
            status_text.text(f"✅ Processing complete!")
            
        # Save knowledge base
        self.save_knowledge_base()
        
        st.success(f"🎉 Successfully processed {files_processed} biology documents with {len(self.knowledge_base)} total chunks!")
        
        return files_processed > 0
    
    def save_knowledge_base(self):
        """Save the biology knowledge base to file"""
        try:
            data = {
                'knowledge_base': self.knowledge_base,
                'metadata': self.metadata,
                'created_at': datetime.now().isoformat(),
                'version': '1.0',
                'domain': 'biology'
            }
            
            with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            st.error(f"Error saving knowledge base: {e}")
    
    def load_knowledge_base(self) -> bool:
        """Load existing biology knowledge base"""
        try:
            if os.path.exists(self.knowledge_file):
                with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.knowledge_base = data.get('knowledge_base', [])
                    self.metadata = data.get('metadata', {})
                    return len(self.knowledge_base) > 0
        except Exception as e:
            st.error(f"Error loading knowledge base: {e}")
        
        return False

class BiologyRetriever:
    """Specialized retrieval system for biology content"""
    
    def __init__(self, knowledge_base: List[Dict]):
        self.knowledge_base = knowledge_base
    
    def calculate_biology_relevance_score(self, query: str, chunk: Dict) -> float:
        """Calculate relevance score optimized for biology queries"""
        query_lower = query.lower()
        content_lower = chunk['content'].lower()
        
        score = 0.0
        
        # Exact phrase matching (highest weight)
        if query_lower in content_lower:
            score += 10.0
        
        # Biology topic matching - boost if query topics match chunk topics
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        chunk_topics = set(chunk.get('topics', []))
        
        # High boost for topic matches
        topic_matches = len(query_words.intersection(chunk_topics))
        if topic_matches > 0:
            score += topic_matches * 3.0
        
        # General word overlap scoring
        content_words = set(re.findall(r'\b\w+\b', content_lower))
        
        if query_words and content_words:
            overlap = len(query_words.intersection(content_words))
            score += (overlap / len(query_words)) * 4.0
        
        # Boost for heart-related content when heart questions are asked
        heart_terms = ['heart', 'cardiac', 'cardiovascular', 'ventricle', 'atrium']
        if any(term in query_lower for term in heart_terms):
            if any(term in content_lower for term in heart_terms):
                score += 3.0
        
        # Boost for anatomy/physiology content
        anatomy_terms = ['anatomy', 'structure', 'physiology', 'function', 'system']
        if any(term in query_lower for term in anatomy_terms):
            if any(term in content_lower for term in anatomy_terms):
                score += 2.0
        
        # Length penalty for very short chunks
        if len(chunk['content']) < 150:
            score *= 0.7
        
        # Boost for chunks with rich biology content
        if len(chunk.get('topics', [])) > 3:
            score *= 1.2
        
        return score
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Retrieve most relevant biology chunks for the query"""
        if not self.knowledge_base:
            return []
        
        # Score all chunks
        scored_chunks = []
        for chunk in self.knowledge_base:
            score = self.calculate_biology_relevance_score(query, chunk)
            if score > 0:  # Only include chunks with some relevance
                scored_chunks.append((chunk, score))
        
        # Sort by score and return top_k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

class BiologyRAGSystem:
    """Complete Biology-focused RAG-LLM system"""
    
    def __init__(self):
        self.processor = BiologyDocumentProcessor()
        self.retriever = None
        self.model_name = "llama2"
        self.conversation_history = []
    
    def initialize_system(self) -> bool:
        """Initialize the biology RAG system"""
        # Try to load existing knowledge base
        if self.processor.load_knowledge_base():
            st.success("🔬 Loaded existing biology knowledge base!")
            self.retriever = BiologyRetriever(self.processor.knowledge_base)
            return True
        else:
            # Process documents
            if self.processor.process_all_documents():
                self.retriever = BiologyRetriever(self.processor.knowledge_base)
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
    
    def create_biology_prompt(self, query: str, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        """Create a biology-focused prompt with context"""
        
        if not relevant_chunks:
            return f"""You are a biology and medical expert AI assistant. The user asked: "{query}"
            
I don't have any relevant biology documents to answer this question. Please let the user know that you need them to provide biology or medical documents, or ask about topics covered in the available knowledge base."""
        
        # Build context from relevant chunks
        context_parts = []
        sources_used = set()
        
        for i, (chunk, score) in enumerate(relevant_chunks, 1):
            source = chunk['source']
            sources_used.add(source)
            topics = ", ".join(chunk.get('topics', [])[:5])  # Show top topics
            
            context_parts.append(f"""
Document {i} (from {source}, relevance: {score:.1f}, topics: {topics}):
{chunk['content']}
""")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are an expert biology and medical AI assistant with access to specialized knowledge about anatomy, physiology, and cardiovascular systems.

CONTEXT FROM BIOLOGY DOCUMENTS:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer the question based primarily on the provided biology context
2. Use proper medical and biological terminology
3. Be comprehensive but clear - explain complex concepts accessibly
4. If the context doesn't fully answer the question, say so clearly
5. Reference which document(s) you're drawing information from
6. When discussing biological processes, explain the mechanisms involved
7. If appropriate, mention related biological concepts or systems
8. Focus on accuracy and scientific rigor

SOURCES AVAILABLE: {', '.join(sources_used)}

Please provide a detailed, scientifically accurate answer:"""
        
        return prompt
    
    def generate_response(self, query: str) -> Dict:
        """Generate response using the biology RAG pipeline"""
        
        if not self.retriever:
            return {
                'answer': "❌ Biology system not initialized. Please process documents first.",
                'sources': [],
                'confidence': 0.0,
                'error': True
            }
        
        try:
            # Step 1: Retrieve relevant biology chunks
            relevant_chunks = self.retriever.retrieve_relevant_chunks(query, top_k=5)
            
            if not relevant_chunks:
                return {
                    'answer': f"I couldn't find any relevant biological information to answer your question about '{query}'. The available documents cover heart anatomy, cardiac physiology, and cardiovascular systems. Please ask about these biology topics or add more relevant documents.",
                    'sources': [],
                    'confidence': 0.0,
                    'error': False
                }
            
            # Step 2: Create biology-focused context prompt
            prompt = self.create_biology_prompt(query, relevant_chunks)
            
            # Step 3: Generate response using Ollama
            try:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'temperature': 0.6,  # Slightly lower for more precise biology answers
                        'top_p': 0.9,
                        'num_predict': 1000  # More tokens for detailed biology explanations
                    }
                )
                
                # Calculate confidence based on relevance scores
                avg_score = sum(score for _, score in relevant_chunks) / len(relevant_chunks)
                confidence = min(avg_score / 12.0, 1.0)  # Normalize to 0-1
                
                return {
                    'answer': response['response'].strip(),
                    'sources': [chunk['content'] for chunk, _ in relevant_chunks],
                    'source_files': [chunk['source'] for chunk, _ in relevant_chunks],
                    'topics': [chunk.get('topics', []) for chunk, _ in relevant_chunks],
                    'confidence': confidence,
                    'error': False
                }
                
            except Exception as e:
                return {
                    'answer': f"❌ Error generating response with {self.model_name}: {str(e)}\n\nTry installing the model with: ollama pull {self.model_name}",
                    'sources': [],
                    'confidence': 0.0,
                    'error': True
                }
                
        except Exception as e:
            return {
                'answer': f"❌ Unexpected error in biology RAG system: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'error': True
            }

def main():
    """Main Streamlit application for biology RAG system"""
    
    st.set_page_config(
        page_title="Biology RAG-LLM System",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Heart RAG-LLM System")
    st.markdown("*Your personal AI assistant for heart-related knowledge*")

    # Initialize session state
    if 'biology_rag_system' not in st.session_state:
        st.session_state.biology_rag_system = BiologyRAGSystem()
        st.session_state.system_initialized = False
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.header("🧬 Cardiovascular")

        # System initialization
        if not st.session_state.system_initialized:
            st.warning("⚠️ System not initialized")
            if st.button("🚀 Initialize RAG System", type="primary"):
                with st.spinner("Processing documents..."):
                    if st.session_state.biology_rag_system.initialize_system():
                        st.session_state.system_initialized = True
                        st.success("✅ System initialized!")
                        st.rerun()
                    else:
                        st.error("Failed to initialize System")
        else:
            st.success("✅ System Ready")
            
            # Reinitialize option
            if st.button("🔄 Reprocess Documents"):
                st.session_state.biology_rag_system.processor.knowledge_base = []
                if st.session_state.biology_rag_system.processor.process_all_documents():
                    st.session_state.biology_rag_system.retriever = BiologyRetriever(
                        st.session_state.biology_rag_system.processor.knowledge_base
                    )
                    st.success("✅ Documents reprocessed!")
                    st.rerun()
        
        st.markdown("---")
        
        # Model selection
        st.subheader("🤖 Model Settings")
        available_models = st.session_state.biology_rag_system.get_available_models()
        selected_model = st.selectbox(
            "Select LLM Model",
            available_models,
            index=available_models.index(st.session_state.biology_rag_system.model_name) 
            if st.session_state.biology_rag_system.model_name in available_models else 0
        )
        
        if selected_model != st.session_state.biology_rag_system.model_name:
            st.session_state.biology_rag_system.model_name = selected_model
            st.success(f"Model changed to {selected_model}")
        
        st.markdown("---")
        
        # Knowledge base info
        st.subheader("📊 Knowledge Base")
        if st.session_state.system_initialized:
            kb_size = len(st.session_state.biology_rag_system.processor.knowledge_base)
            st.metric("Total Chunks", kb_size)
            
            # Show document breakdown and topics
            if st.session_state.biology_rag_system.processor.metadata:
                with st.expander("📋 Document Details"):
                    for filename, info in st.session_state.biology_rag_system.processor.metadata.items():
                        st.write(f"**{filename}**")
                        st.write(f"- Chunks: {info['chunks_count']}")
                        st.write(f"- Type: {info['file_type'].upper()}")
                        st.write(f"- Size: {info['file_size']:,} chars")
                        if info.get('topics_found'):
                            st.write(f"- Key Topics: {', '.join(info['topics_found'][:8])}")
        else:
            st.metric("Total Biology Chunks", "Not initialized")
    
    # Main chat interface
    if st.session_state.system_initialized:
        st.header("💬 Ask Cardiovascular Questions")

        # Display conversation history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant" and "sources" in message:
                    if message.get("sources"):
                        with st.expander("📚 View Cardiovascular Sources & References"):
                            for i, (source, source_file, topics) in enumerate(zip(
                                message["sources"][:3], 
                                message.get("source_files", []), 
                                message.get("topics", [])
                            ), 1):
                                st.markdown(f"**Source {i} (from {source_file}):**")
                                if topics:
                                    st.markdown(f"*Topics: {', '.join(topics[:5])}*")
                                st.text_area(
                                    f"Cardiovascular Content {i}",
                                    source[:500] + "..." if len(source) > 500 else source,
                                    height=180,
                                    key=f"bio_source_{message.get('timestamp', '')}_{i}"
                                )
        
        # Chat input
        if prompt := st.chat_input("Ask about heart anatomy, cardiovascular physiology..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🔬 Analyzing cardiovascular content..."):
                    result = st.session_state.biology_rag_system.generate_response(prompt)
                
                st.markdown(result["answer"])
                
                # Show confidence and metadata
                if not result.get("error", False):
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        confidence = result.get("confidence", 0)
                        color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
                        st.markdown(f"**Confidence:** {color} {confidence:.2f}")
                    
                    with col2:
                        if result.get("source_files"):
                            sources_text = ", ".join(set(result["source_files"]))
                            st.markdown(f"**Sources:** {sources_text}")
                
                # Add sources to message for expandable view
                message_data = {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "source_files": result.get("source_files", []),
                    "topics": result.get("topics", []),
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(message_data)
        
        # Cardiovascular-specific example questions
        st.markdown("---")
        st.markdown("### 💡 Cardiovascular Questions to Try:")

        cardiovascular_questions = [
            "What is the structure of the human heart?",
            "How does blood circulation work?",
            "What are the main functions of the cardiovascular system?",
            "Explain the anatomy of heart chambers and valves",
            "How does the heart pump blood through the body?",
            "What is the difference between arteries and veins?",
            "Describe the cardiac cycle and heart rhythm",
            "What role do heart valves play in circulation?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(cardiovascular_questions):
            with cols[i % 2]:
                if st.button(question, key=f"bio_example_{i}"):
                    st.session_state.messages.append({"role": "user", "content": question})
                    with st.spinner("🔬 Analyzing cardiovascular content..."):
                        result = st.session_state.biology_rag_system.generate_response(question)
                    
                    message_data = {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                        "source_files": result.get("source_files", []),
                        "topics": result.get("topics", []),
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.messages.append(message_data)
                    st.rerun()
    
    else:
        # System not initialized
        st.info("🚀 Click 'Initialize Biology RAG System' in the sidebar to get started!")
        
        st.markdown("""
        ### 🔬 Your Biology Knowledge Base:
        This system will process the biology documents in your `data` folder:
        - **Heart Biology**: Anatomy and physiology chapters
        - **Cardiovascular System**: Structure and function
        - **Medical Content**: Related biological processes
        
        ### 🎯 What You Can Do:
        - Ask detailed questions about heart anatomy and physiology
        - Explore cardiovascular system functions
        - Learn about biological processes and mechanisms
        - Get scientifically accurate answers with source references
        - Understand complex biology concepts with clear explanations
        """)

if __name__ == "__main__":
    main()