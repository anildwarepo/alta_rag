# %%
import os
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import openai_helper

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "__dataclass_fields__"):  # Check if it's a dataclass
            return asdict(obj)
        return super().default(obj)


@dataclass
class Element:
    """Base class for all elements."""
    id: Optional[str] = None

@dataclass
class Title(Element):
    content: str = ''

@dataclass
class Subtitle(Element):
    content: str = ''


@dataclass
class Xref(Element):
    xreflabel: str = ''
    linkend: str = ''

@dataclass
class Para(Element):
    content: str = ''
    #xref: Xref = None


@dataclass
class Section(Element):
    title: Optional[Title] = None
    paras: List[Para] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    xrefs: List[Xref] = field(default_factory=list)

@dataclass
class Preface(Element):
    title: Optional[Title] = None
    sections: List[Section] = field(default_factory=list)

@dataclass
class Appendix(Element):
    title: Optional[Title] = None
    sections: List[Section] = field(default_factory=list)

@dataclass
class Chapter(Element):
    title: Optional[Title] = None
    sections: List[Section] = field(default_factory=list)

@dataclass
class Part(Element):
    title: Optional[Title] = None
    chapters: List['Chapter'] = field(default_factory=list)

@dataclass
class Book(Element):
    title: Optional[Title] = None
    subtitle: Optional[Subtitle] = None
    preface: Optional[Preface] = None
    chapters: List['Chapter'] = field(default_factory=list)
    appendix : List['Appendix'] = field(default_factory=list)
    parts: List['Part'] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)



# %%
def serialize_book_to_json(book: Book) -> str:
    """Serialize the book object to a JSON-formatted string."""
    # Convert the dataclass instance to a dictionary
    book_dict = asdict(book)
    # Serialize the dictionary to a JSON string
    json_str = json.dumps(book_dict, indent=2)
    return json_str

def get_title_id(node):
    title = ''
    if node.find('title') is not None:
        title = node.find('title').text
    title_id = node.attrib.get('id')
    return title, title_id

def process_chapters(part_node, book_part):
    
        for section in part_node.findall('.//section'):
            section_title, section_id = get_title_id(section)
            book_section = Section(
                title=Title(content=section_title),
                id = section_id,
                paras=[],
                keywords=[],
            )
            book_part.sections.append(book_section)
            

            for xrefnode in section.findall('.//xref'):
                xref = Xref(xreflabel=xrefnode.attrib.get('xreflabel'), linkend=xrefnode.attrib.get('linkend'))
                book_section.xrefs.append(xref)

            for para in section.findall('.//para'):
                #xrefnode = para.find('.//xref')
                
                #xref = None
                #if xrefnode is not None:
                #    xref = Xref(xreflabel=xrefnode.attrib.get('xreflabel'), linkend=xrefnode.attrib.get('linkend'))
                book_para = Para(
                    content= ''.join(para.itertext()),#para.text,                    
                    #xref=xref
                )
                book_section.paras.append(book_para)
            for itemlist in section.findall('.//itemizedlist'):
                book_para = Para(
                    content= ''.join(itemlist.itertext()) #itemizedlist.text,
                )
                book_section.paras.append(book_para)
            keyword_nodes = section.findall('.//keyword')
            if keyword_nodes:
                for keyword in keyword_nodes:
                    if book_section.keywords.count(keyword.text) == 0:
                        book_section.keywords.append(keyword.text)

book_chunks = []
def process_book(root, parts, chapters, appendix):

    book_info = {}


    for part in parts:
        print(len(book_chunks))
        book_info["book_title"] = get_title_id(root)[0]
        book_info["part_title"] = part.title.content
        book_info["part_id"] = part.id
        for chapter in part.chapters:
            book_info["chapter_title"] = chapter.title.content
            book_info["chapter_id"] = chapter.id
            for section in chapter.sections:
                book_info["section_title"] = section.title.content
                book_info["section_id"] = section.id
                book_info["keywords"] = section.keywords
                book_info["xrefs"] = json.dumps(section.xrefs, cls=CustomEncoder)

                para_chunk = ""
                for para in section.paras:
                    #book_info["para"] = para.content
                    #llm_context += json.dumps(book_info) + "\n"
                    if para.content is None:
                        continue

                    para_chunk += para.content
                    token_count = openai_helper.get_token_count(para_chunk)
                    #book_info["token_count"] = token_count
                    #print(token_count)
                    #print(book_info)
                    if token_count > 500:
                        book_info["para"] = para_chunk
                        book_chunks.append(json.dumps(book_info))
                        para_chunk = ""
                        continue
                if para_chunk:
                    book_info["para"] = para_chunk
                    book_chunks.append(json.dumps(book_info))



    if chapters:
        book_info = {}
        book_info["book_title"] = get_title_id(root)[0]
        for chapter in chapters:
            book_info["chapter_title"] = chapter.title.content
            book_info["chapter_id"] = chapter.id
            for section in chapter.sections:
                book_info["section_title"] = section.title.content
                book_info["section_id"] = section.id
                book_info["keywords"] = section.keywords
                book_info["xrefs"] = json.dumps(section.xrefs, cls=CustomEncoder)
                para_chunk = ""
                for para in section.paras:
                    #book_info["para"] = para.content
                    #llm_context += json.dumps(book_info) + "\n"
                    if para.content is None:
                        continue

                    #book_info["xreflabel"] = para.xref.xreflabel if para.xref else ""
                    #book_info["linkend"] = para.xref.linkend if para.xref else ""

                    para_chunk += para.content
                    token_count = openai_helper.get_token_count(para_chunk)
                    #book_info["token_count"] = token_count
                    #print(token_count)
                    #print(book_info)
                    if token_count > 500:
                        book_info["para"] = para_chunk
                        book_chunks.append(json.dumps(book_info))
                        para_chunk = ""
                        continue
                if para_chunk:
                    book_info["para"] = para_chunk
                    book_chunks.append(json.dumps(book_info))

    if appendix:
        book_info = {}
        book_info["book_title"] = get_title_id(root)[0]
        for appendix in appendix:
            book_info["appendix_title"] = appendix.title.content
            book_info["appendix_id"] = appendix.id
            for section in appendix.sections:
                book_info["section_title"] = section.title.content
                book_info["section_id"] = section.id
                book_info["keywords"] = section.keywords
                book_info["xrefs"] = json.dumps(section.xrefs, cls=CustomEncoder)
                para_chunk = ""
                for para in section.paras:
                    #book_info["para"] = para.content
                    #llm_context += json.dumps(book_info) + "\n"
                    if para.content is None:
                        continue
                    para_chunk += para.content
                    token_count = openai_helper.get_token_count(para_chunk)
                    #book_info["token_count"] = token_count
                    #print(token_count)
                    #print(book_info)
                    if token_count > 500:
                        book_info["para"] = para_chunk
                        book_chunks.append(json.dumps(book_info))
                        para_chunk = ""
                        continue
                if para_chunk:
                    book_info["para"] = para_chunk
                    book_chunks.append(json.dumps(book_info))

def write_to_file():
    with open("processed_book_chunks.json", "w") as file:
        file.write(json.dumps(book_chunks, indent=2))



# %% [markdown]
# NBU_ADMIN1_nbuadmin1-uma.xml
# 
#     part
#     chapter
#     section
# 
# NBU_DB2_db2_uma.xml
# 
#     chapter
#     section
# 

# %%
def process_nodes(root, part_nodes, chapter_nodes, appendix_nodes):
    parts = []
    chapters = []
    appendix = []

    if part_nodes:
        for childnode in part_nodes:
            part_title, part_id = get_title_id(childnode)
            book_part = Part(
                title=Title(content=part_title),
                id = part_id,
                chapters=[],
            )
            parts.append(book_part)
            for chapter in childnode.findall('chapter'):
                chapter_title, chapter_id = get_title_id(chapter)
                book_chapter = Chapter(
                    title=Title(content=chapter_title),
                    id=chapter_id,
                    sections=[],
                )
                book_part.chapters.append(book_chapter)
                process_chapters(chapter, book_chapter)

    if appendix_nodes:

        for childnode in appendix_nodes:
            title, id = get_title_id(childnode)
            book_appendix = Appendix(
                    title=Title(content=title),
                    id=id,
                    sections=[],
            )
            appendix.append(book_appendix)
            process_chapters(childnode, book_appendix)


    if chapter_nodes:

        for childnode in chapter_nodes:
            title, id = get_title_id(childnode)
            book_chapter = Chapter(
                    title=Title(content=title),
                    id=id,
                    sections=[],
            )
            chapters.append(book_chapter)
            process_chapters(childnode, book_chapter)



    process_book(root, parts, chapters, appendix)

    print(parts)
    print(chapters)
    print(appendix)



# %%
rag_docs_path = "Rag Optimization"

for file in os.listdir(rag_docs_path):
    if file.endswith(".xml"):
        xml_file_path = os.path.join(rag_docs_path, file)
        with open(xml_file_path, 'r', encoding='utf-8') as file:
            xml_content = file.read().replace('&nbsp;', ' ') 
        root = ET.fromstring(xml_content)
        part_nodes = root.findall('part')
        chapter_nodes = root.findall('chapter')
        appendix_nodes = root.findall('appendix')
        process_nodes(root, part_nodes, chapter_nodes, appendix_nodes)
       
write_to_file()



