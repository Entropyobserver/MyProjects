import torch
import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import re
import json
import csv
import io
import tempfile
import os
from datetime import datetime

# Configuration
MAX_HISTORY_SIZE = 1000
BATCH_SIZE_LIMIT = 50
THEMES = {
    'default': {'pos': '#4ecdc4', 'neg': '#ff6b6b'},
    'ocean': {'pos': '#0077be', 'neg': '#ff6b35'},
    'forest': {'pos': '#228b22', 'neg': '#dc143c'},
    'sunset': {'pos': '#ff8c00', 'neg': '#8b0000'}
}

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("entropy25/sentimentanalysis")
model = BertForSequenceClassification.from_pretrained("entropy25/sentimentanalysis")
model.to(device)

# Global storage with size limit
history = []

def manage_history_size():
    """Keep history size under limit"""
    global history
    if len(history) > MAX_HISTORY_SIZE:
        history = history[-MAX_HISTORY_SIZE:]

def clean_text(text):
    """Simple text preprocessing"""
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    return [w for w in words if w not in stopwords and len(w) > 2]

def analyze_text(text, theme='default'):
    """Core sentiment analysis"""
    if not text.strip():
        return "Please enter text", None, None, None
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred = torch.argmax(outputs.logits, dim=-1).item()
        conf = probs.max()
        sentiment = "Positive" if pred == 1 else "Negative"
    
    # Store in history with timestamp
    history.append({
        'text': text[:100],
        'full_text': text,
        'sentiment': sentiment,
        'confidence': conf,
        'pos_prob': probs[1],
        'neg_prob': probs[0],
        'timestamp': datetime.now().isoformat()
    })
    
    manage_history_size()
    
    result = f"Sentiment: {sentiment} (Confidence: {conf:.3f})"
    
    # Generate plots
    prob_plot = plot_probs(probs, theme)
    gauge_plot = plot_gauge(conf, sentiment, theme)
    cloud_plot = plot_wordcloud(text, sentiment, theme)
    
    return result, prob_plot, gauge_plot, cloud_plot

def plot_probs(probs, theme='default'):
    """Probability bar chart"""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Negative", "Positive"]
    colors = [THEMES[theme]['neg'], THEMES[theme]['pos']]
    
    bars = ax.bar(labels, probs, color=colors, alpha=0.8)
    ax.set_title("Sentiment Probabilities", fontweight='bold')
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_gauge(conf, sentiment, theme='default'):
    """Confidence gauge"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    theta = np.linspace(0, np.pi, 100)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 100))
    
    for i in range(len(theta)-1):
        ax.fill_between([theta[i], theta[i+1]], [0, 0], [0.8, 0.8], 
                       color=colors[i], alpha=0.7)
    
    pos = np.pi * (0.5 + (0.4 if sentiment == 'Positive' else -0.4) * conf)
    ax.plot([pos, pos], [0, 0.6], 'k-', linewidth=6)
    ax.plot(pos, 0.6, 'ko', markersize=10)
    
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 1)
    ax.set_title(f'{sentiment} - Confidence: {conf:.3f}', fontweight='bold')
    ax.set_xticks([0, np.pi/2, np.pi])
    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_yticks([])
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def plot_wordcloud(text, sentiment, theme='default'):
    """Word cloud visualization"""
    if len(text.split()) < 3:
        return None
    
    colormap = 'Greens' if sentiment == 'Positive' else 'Reds'
    wc = WordCloud(width=800, height=400, background_color='white',
                  colormap=colormap, max_words=30).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'{sentiment} Word Cloud', fontweight='bold')
    
    plt.tight_layout()
    return fig

def batch_analysis(reviews, progress=gr.Progress()):
    """Analyze multiple reviews with progress tracking"""
    if not reviews.strip():
        return None
    
    texts = [r.strip() for r in reviews.split('\n') if r.strip()]
    if len(texts) < 2:
        return None
    
    # Apply batch size limit
    if len(texts) > BATCH_SIZE_LIMIT:
        texts = texts[:BATCH_SIZE_LIMIT]
    
    results = []
    
    for i, text in enumerate(texts):
        progress((i + 1) / len(texts), f"Processing review {i + 1}/{len(texts)}")
        
        # Process in smaller GPU batches
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            pred = torch.argmax(outputs.logits, dim=-1).item()
            sentiment = "Positive" if pred == 1 else "Negative"
            conf = probs.max()
            
        results.append({
            'text': text[:50] + '...' if len(text) > 50 else text,
            'sentiment': sentiment,
            'confidence': conf,
            'pos_prob': probs[1]
        })
        
        # Add to history
        history.append({
            'text': text[:100],
            'full_text': text,
            'sentiment': sentiment,
            'confidence': conf,
            'pos_prob': probs[1],
            'neg_prob': probs[0],
            'timestamp': datetime.now().isoformat()
        })
    
    manage_history_size()
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Pie chart
    sent_counts = Counter([r['sentiment'] for r in results])
    colors = ['#4ecdc4', '#ff6b6b']
    ax1.pie(sent_counts.values(), labels=sent_counts.keys(), 
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Sentiment Distribution')
    
    # Confidence histogram
    confs = [r['confidence'] for r in results]
    ax2.hist(confs, bins=8, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('Confidence Distribution')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    
    # Probability scatter
    indices = range(len(results))
    pos_probs = [r['pos_prob'] for r in results]
    ax3.scatter(indices, pos_probs, 
               c=['#4ecdc4' if r['sentiment'] == 'Positive' else '#ff6b6b' for r in results],
               alpha=0.7, s=100)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.set_title('Positive Probability by Review')
    ax3.set_xlabel('Review Index')
    ax3.set_ylabel('Positive Probability')
    
    # Confidence vs Sentiment
    sent_binary = [1 if r['sentiment'] == 'Positive' else 0 for r in results]
    ax4.scatter(confs, sent_binary, alpha=0.7, s=100, 
               c=['#4ecdc4' if s == 1 else '#ff6b6b' for s in sent_binary])
    ax4.set_title('Sentiment vs Confidence')
    ax4.set_xlabel('Confidence')
    ax4.set_ylabel('Sentiment')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Negative', 'Positive'])
    
    plt.tight_layout()
    return fig

def process_uploaded_file(file):
    """Process uploaded CSV/TXT file for batch analysis"""
    if file is None:
        return ""
    
    content = file.read().decode('utf-8')
    
    # Handle CSV format
    if file.name.endswith('.csv'):
        lines = content.split('\n')
        # Assume text is in first column or look for 'review' column
        if ',' in content:
            reviews = []
            reader = csv.reader(lines)
            headers = next(reader, None)
            if headers and any('review' in h.lower() for h in headers):
                review_idx = next(i for i, h in enumerate(headers) if 'review' in h.lower())
                for row in reader:
                    if len(row) > review_idx:
                        reviews.append(row[review_idx])
            else:
                for row in reader:
                    if row:
                        reviews.append(row[0])
            return '\n'.join(reviews)
    
    # Handle plain text
    return content

def export_history_csv():
    """Export history to CSV file"""
    if not history:
        return None, "No history to export"
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8')
    writer = csv.writer(temp_file)
    writer.writerow(['Timestamp', 'Text', 'Sentiment', 'Confidence', 'Positive_Prob', 'Negative_Prob'])
    
    for entry in history:
        writer.writerow([
            entry['timestamp'], entry['text'], entry['sentiment'],
            f"{entry['confidence']:.4f}", f"{entry['pos_prob']:.4f}", f"{entry['neg_prob']:.4f}"
        ])
    
    temp_file.close()
    return temp_file.name, f"Exported {len(history)} entries to CSV"

def export_history_json():
    """Export history to JSON file"""
    if not history:
        return None, "No history to export"
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8')
    json.dump(history, temp_file, indent=2, ensure_ascii=False)
    temp_file.close()
    
    return temp_file.name, f"Exported {len(history)} entries to JSON"

def keyword_heatmap():
    """Keyword sentiment heatmap"""
    if len(history) < 3:
        return None
    
    word_stats = defaultdict(list)
    
    for item in history:
        words = clean_text(item['full_text'])
        sentiment_score = item['pos_prob']
        
        for word in words:
            word_stats[word].append(sentiment_score)
    
    # Filter words with at least 2 occurrences
    filtered = {w: scores for w, scores in word_stats.items() if len(scores) >= 2}
    
    if len(filtered) < 5:
        return None
    
    # Get top 20 most frequent words
    top_words = sorted(filtered.items(), key=lambda x: len(x[1]), reverse=True)[:20]
    
    words = [item[0] for item in top_words]
    avg_sentiments = [np.mean(item[1]) for item in top_words]
    frequencies = [len(item[1]) for item in top_words]
    
    # Create heatmap data
    data = np.array([avg_sentiments, [f/max(frequencies) for f in frequencies]]).T
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Avg Sentiment', 'Frequency'])
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    
    # Add text annotations
    for i in range(len(words)):
        ax.text(0, i, f'{avg_sentiments[i]:.2f}', ha='center', va='center', 
                color='black', fontweight='bold')
        ax.text(1, i, f'{frequencies[i]}', ha='center', va='center', 
                color='black', fontweight='bold')
    
    ax.set_title('Keyword Sentiment Heatmap', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Intensity')
    
    plt.tight_layout()
    return fig

def cooccurrence_network():
    """Word co-occurrence network"""
    if len(history) < 3:
        return None
    
    all_words = []
    for item in history:
        words = clean_text(item['full_text'])
        if len(words) >= 3:
            all_words.extend(words)
    
    if len(all_words) < 10:
        return None
    
    word_freq = Counter(all_words)
    top_words = [word for word, freq in word_freq.most_common(15) if freq >= 2]
    
    if len(top_words) < 5:
        return None
    
    # Calculate co-occurrences
    cooccur = defaultdict(int)
    
    for item in history:
        words = [w for w in clean_text(item['full_text']) if w in top_words]
        
        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words):
                if i != j and w1 != w2:
                    pair = tuple(sorted([w1, w2]))
                    cooccur[pair] += 1
    
    # Create network
    G = nx.Graph()
    
    for word in top_words:
        G.add_node(word, size=word_freq[word])
    
    for (w1, w2), weight in cooccur.items():
        if weight >= 2:
            G.add_edge(w1, w2, weight=weight)
    
    if len(G.edges()) == 0:
        return None
    
    # Plot network
    fig, ax = plt.subplots(figsize=(12, 10))
    
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    node_sizes = [G.nodes[node]['size'] * 200 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.7, ax=ax)
    
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w*0.5 for w in weights], 
                          alpha=0.6, edge_color='gray', ax=ax)
    
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    ax.set_title('Word Co-occurrence Network', fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def tfidf_analysis():
    """TF-IDF keyword analysis"""
    if len(history) < 4:
        return None
    
    pos_texts = []
    neg_texts = []
    
    for item in history:
        if item['sentiment'] == 'Positive':
            pos_texts.append(' '.join(clean_text(item['full_text'])))
        else:
            neg_texts.append(' '.join(clean_text(item['full_text'])))
    
    if len(pos_texts) < 2 or len(neg_texts) < 2:
        return None
    
    # Positive TF-IDF
    vectorizer_pos = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
    pos_tfidf = vectorizer_pos.fit_transform(pos_texts)
    pos_features = vectorizer_pos.get_feature_names_out()
    pos_scores = pos_tfidf.sum(axis=0).A1
    
    # Negative TF-IDF
    vectorizer_neg = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
    neg_tfidf = vectorizer_neg.fit_transform(neg_texts)
    neg_features = vectorizer_neg.get_feature_names_out()
    neg_scores = neg_tfidf.sum(axis=0).A1
    
    # Top 10 features
    pos_top_idx = np.argsort(pos_scores)[-10:][::-1]
    neg_top_idx = np.argsort(neg_scores)[-10:][::-1]
    
    pos_words = [pos_features[i] for i in pos_top_idx]
    pos_vals = [pos_scores[i] for i in pos_top_idx]
    
    neg_words = [neg_features[i] for i in neg_top_idx]
    neg_vals = [neg_scores[i] for i in neg_top_idx]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Positive
    bars1 = ax1.barh(pos_words, pos_vals, color='#4ecdc4', alpha=0.8)
    ax1.set_title('Positive Keywords (TF-IDF)', fontweight='bold')
    ax1.set_xlabel('TF-IDF Score')
    
    for bar, score in zip(bars1, pos_vals):
        ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=9)
    
    # Negative
    bars2 = ax2.barh(neg_words, neg_vals, color='#ff6b6b', alpha=0.8)
    ax2.set_title('Negative Keywords (TF-IDF)', fontweight='bold')
    ax2.set_xlabel('TF-IDF Score')
    
    for bar, score in zip(bars2, neg_vals):
        ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_history():
    """Analysis history visualization"""
    if len(history) < 2:
        return None, f"History contains {len(history)} entries. Need at least 2 for visualization."
    
    indices = list(range(len(history)))
    pos_probs = [item['pos_prob'] for item in history]
    confs = [item['confidence'] for item in history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    colors = ['#4ecdc4' if p > 0.5 else '#ff6b6b' for p in pos_probs]
    ax1.scatter(indices, pos_probs, c=colors, alpha=0.7, s=100)
    ax1.plot(indices, pos_probs, alpha=0.5, linewidth=2)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Sentiment History - Positive Probability')
    ax1.set_xlabel('Analysis Number')
    ax1.set_ylabel('Positive Probability')
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(indices, confs, alpha=0.7, color='lightblue', edgecolor='navy')
    ax2.set_title('Confidence Scores Over Time')
    ax2.set_xlabel('Analysis Number')
    ax2.set_ylabel('Confidence Score')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, f"History contains {len(history)} analyses"

def clear_history():
    """Clear analysis history"""
    global history
    count = len(history)
    history.clear()
    return f"Cleared {count} entries from history"

def get_history_status():
    """Get current history status"""
    return f"History contains {len(history)} entries"

# Enhanced example data
EXAMPLE_REVIEWS = [
    ["The cinematography was stunning, but the plot felt predictable and the dialogue was weak."],
    ["A masterpiece of filmmaking! Amazing performances, brilliant direction, and unforgettable moments."],
    ["Boring movie with terrible acting, weak plot, and poor character development throughout."],
    ["Great special effects and action sequences, but the story was confusing and hard to follow."],
    ["Incredible ending that left me speechless! One of the best films I've ever seen."],
    ["The movie started strong but became repetitive and lost my interest halfway through."],
    ["Outstanding soundtrack and beautiful visuals, though the pacing was somewhat slow."],
    ["Disappointing sequel that failed to capture the magic of the original film."],
    ["Brilliant writing and exceptional acting make this a must-watch drama."],
    ["Generic blockbuster with predictable twists and forgettable characters."]
]

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Movie Sentiment Analyzer") as demo:
    gr.Markdown("# 🎬 AI Movie Sentiment Analyzer")
    gr.Markdown("Advanced sentiment analysis with comprehensive visualizations and data export capabilities")
    
    with gr.Tab("Single Analysis"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Movie Review",
                    placeholder="Enter your movie review here...",
                    lines=5
                )
                
                with gr.Row():
                    analyze_btn = gr.Button("Analyze", variant="primary", size="lg")
                    theme_selector = gr.Dropdown(
                        choices=list(THEMES.keys()),
                        value="default",
                        label="Color Theme"
                    )
                
                gr.Examples(
                    examples=EXAMPLE_REVIEWS,
                    inputs=text_input,
                    label="Example Reviews"
                )
            
            with gr.Column():
                result_output = gr.Textbox(label="Analysis Result", lines=2)
        
        with gr.Row():
            prob_plot = gr.Plot(label="Sentiment Probabilities")
            gauge_plot = gr.Plot(label="Confidence Gauge")
        
        wordcloud_plot = gr.Plot(label="Word Cloud Visualization")
    
    with gr.Tab("Batch Analysis"):
        gr.Markdown("### Multiple Reviews Analysis")
        gr.Markdown(f"**Note:** Limited to {BATCH_SIZE_LIMIT} reviews per batch for optimal performance")
        
        with gr.Row():
            with gr.Column():
                file_upload = gr.File(
                    label="Upload CSV/TXT File",
                    file_types=[".csv", ".txt"],
                    type="binary"
                )
                batch_input = gr.Textbox(
                    label="Reviews (one per line)",
                    placeholder="Review 1...\nReview 2...\nReview 3...",
                    lines=8
                )
            
            with gr.Column():
                load_file_btn = gr.Button("Load File", variant="secondary")
                batch_btn = gr.Button("Analyze Batch", variant="primary")
        
        batch_plot = gr.Plot(label="Batch Analysis Results")
    
    with gr.Tab("Advanced Analytics"):
        gr.Markdown("### Advanced Visualizations")
        gr.Markdown("**Requirements:** Minimum analysis history needed for each visualization")
        
        with gr.Row():
            heatmap_btn = gr.Button("Keyword Heatmap", variant="primary")
            network_btn = gr.Button("Word Network", variant="primary")
            tfidf_btn = gr.Button("TF-IDF Analysis", variant="primary")
        
        heatmap_plot = gr.Plot(label="Keyword Sentiment Heatmap")
        network_plot = gr.Plot(label="Word Co-occurrence Network")
        tfidf_plot = gr.Plot(label="TF-IDF Keywords Comparison")
    
    with gr.Tab("History & Export"):
        gr.Markdown("### Analysis History & Data Export")
        
        with gr.Row():
            refresh_btn = gr.Button("Refresh History", variant="secondary")
            clear_btn = gr.Button("Clear History", variant="stop")
            status_btn = gr.Button("Check Status", variant="secondary")
        
        with gr.Row():
            export_csv_btn = gr.Button("Export CSV", variant="secondary")
            export_json_btn = gr.Button("Export JSON", variant="secondary")
        
        history_status = gr.Textbox(label="Status", interactive=False)
        history_plot = gr.Plot(label="Historical Analysis Trends")
        
        # File downloads
        csv_file_output = gr.File(label="Download CSV", visible=True)
        json_file_output = gr.File(label="Download JSON", visible=True)
    
    # Event handlers
    analyze_btn.click(
        analyze_text,
        inputs=[text_input, theme_selector],
        outputs=[result_output, prob_plot, gauge_plot, wordcloud_plot]
    )
    
    load_file_btn.click(
        process_uploaded_file,
        inputs=file_upload,
        outputs=batch_input
    )
    
    batch_btn.click(
        batch_analysis,
        inputs=batch_input,
        outputs=batch_plot
    )
    
    heatmap_btn.click(keyword_heatmap, outputs=heatmap_plot)
    network_btn.click(cooccurrence_network, outputs=network_plot)
    tfidf_btn.click(tfidf_analysis, outputs=tfidf_plot)
    
    refresh_btn.click(
        plot_history, 
        outputs=[history_plot, history_status]
    )
    
    clear_btn.click(
        clear_history, 
        outputs=history_status
    )
    
    status_btn.click(
        get_history_status,
        outputs=history_status
    )
    
    export_csv_btn.click(
        export_history_csv,
        outputs=[csv_file_output, history_status]
    )
    
    export_json_btn.click(
        export_history_json,
        outputs=[json_file_output, history_status]
    )

if __name__ == "__main__":
    demo.launch(share=True)

