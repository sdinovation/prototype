
import { create } from "zustand";
import { Document } from "@/types/document";
import { mockDocuments } from "@/data/mockData";

interface DocumentStore {
  documents: Document[];
  pinnedDocuments: Document[];
  myDocuments: Document[];
  togglePin: (docId: string) =&gt; void;
  toggleFavorite: (docId: string) =&gt; void;
  deleteDocument: (docId: string) =&gt; void;
}

export const useDocumentStore = create&lt;DocumentStore&gt;((set, get) =&gt; ({
  documents: mockDocuments,
  pinnedDocuments: mockDocuments.filter(d =&gt; d.isPinned),
  myDocuments: [...mockDocuments],

  togglePin: (docId: string) =&gt; {
    set((state) =&gt; {
      const updatedDocs = state.documents.map(doc =&gt; 
        doc.id === docId ? { ...doc, isPinned: !doc.isPinned } : doc
      );
      return {
        documents: updatedDocs,
        pinnedDocuments: updatedDocs.filter(d =&gt; d.isPinned),
        myDocuments: updatedDocs
      };
    });
  },

  toggleFavorite: (docId: string) =&gt; {
    set((state) =&gt; {
      const updatedDocs = state.documents.map(doc =&gt; 
        doc.id === docId ? { ...doc, isFavorite: !doc.isFavorite } : doc
      );
      return {
        documents: updatedDocs,
        pinnedDocuments: updatedDocs.filter(d =&gt; d.isPinned),
        myDocuments: updatedDocs
      };
    });
  },

  deleteDocument: (docId: string) =&gt; {
    set((state) =&gt; {
      const updatedDocs = state.documents.filter(doc =&gt; doc.id !== docId);
      return {
        documents: updatedDocs,
        pinnedDocuments: updatedDocs.filter(d =&gt; d.isPinned),
        myDocuments: updatedDocs
      };
    });
  }
}));
