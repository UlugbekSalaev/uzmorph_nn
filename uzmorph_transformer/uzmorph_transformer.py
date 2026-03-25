import torch
import json
import os
import pkg_resources
from .model import MorphTransformer

class MorphResult:
    """
    A class to represent the morphological analysis result.
    Provides methods to export as dictionary, JSON, and styled string.
    """
    def __init__(self, word, stem, pos, features):
        self.word = word
        self.stem = stem
        self.pos = pos
        self.features = features # list of strings

    def to_dict(self):
        """Returns result as a flattened dictionary."""
        res = {
            "word": self.word,
            "stem": self.stem,
            "pos": self.pos
        }
        # Flatten features (e.g., "possession=1" -> "possession": "1")
        for feat in self.features:
            if '=' in feat:
                k, v = feat.split('=', 1)
                res[k] = v
            else:
                res[feat] = True # For boolean-like features
        return res

    def to_json(self, indent=2):
        """Returns result as a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def __repr__(self):
        return f"MorphResult(word='{self.word}', stem='{self.stem}', pos='{self.pos}', features={self.features})"

    def __str__(self):
        """Returns a human-readable styled string."""
        feat_str = ", ".join(self.features) if self.features else "none"
        return f"Result: '{self.word}' -> Stem: {self.stem} | POS: {self.pos} | Tags: [{feat_str}]"

class uzmorph_transformer:
    def __init__(self, model_path=None, char_vocab=None, tag_vocab=None):
        # Default paths pointing to package data
        if model_path is None:
            model_path = pkg_resources.resource_filename('uzmorph_transformer', 'data/morph_transformer.pth')
        if char_vocab is None:
            char_vocab = pkg_resources.resource_filename('uzmorph_transformer', 'data/char_vocab.json')
        if tag_vocab is None:
            tag_vocab = pkg_resources.resource_filename('uzmorph_transformer', 'data/tag_vocab.json')

        with open(char_vocab, 'r', encoding='utf-8') as f:
            self.char_to_idx = json.load(f)
        with open(tag_vocab, 'r', encoding='utf-8') as f:
            self.tag_to_idx = json.load(f)
        
        self.idx_to_tag = {idx: tag for tag, idx in self.tag_to_idx.items()}
        self.max_len = 30
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = MorphTransformer(len(self.char_to_idx), len(self.tag_to_idx))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def analyze(self, word):
        """
        Analyzes a single word and returns a MorphResult object.
        """
        word = word.strip().lower()
        if not word:
            return MorphResult(word, "", "UNKNOWN", [])
            
        chars = list(word)[:self.max_len]
        indices = [self.char_to_idx.get(c, self.char_to_idx['UNK']) for c in chars]
        
        # Padding
        padded_indices = indices + [self.char_to_idx['PAD']] * (self.max_len - len(indices))
        input_tensor = torch.tensor([padded_indices]).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            predictions = torch.argmax(output, dim=-1)[0]
            
        tags = [self.idx_to_tag.get(idx.item()) for idx in predictions[:len(chars)]]
        
        # Decoding
        stem = ""
        pos = "UNKNOWN"
        features = []
        
        for char, tag in zip(chars, tags):
            if "ROOT" in tag:
                stem += char
            elif "B-SUFF" in tag:
                parts = tag.split('-')
                if len(parts) >= 3:
                    pos = parts[2]
                if len(parts) >= 4:
                    features = parts[3].split('|')
                    
        return MorphResult(word, stem, pos, features)
