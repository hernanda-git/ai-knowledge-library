export interface DocSection {
  title: string;
  content: string;
}

export interface KnowledgeDoc {
  id: string;
  title: string;
  description: string;
  lines: number;
  tags: string[];
  sections: DocSection[];
}

export interface TopicCategory {
  id: string;
  title: string;
  icon: string;
  description: string;
  totalLines: number;
  docCount: number;
  color: string;
}
