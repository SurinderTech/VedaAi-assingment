"""
Comprehensive Regression tests for Post-VLM Corruption Fixes:
Requirement Test Matrix:
A. Same visual entity crossing decomposition boundary -> merged into one canonical structure.
B. Different entities near the decomposition boundary -> NOT merged.
C. Relationships from both crop-local representations -> remapped to canonical IDs.
D. Duplicate relationships -> deduplicated.
E. Self-loop relationships created by remapping -> dropped.
F. Metadata, reasoning, confidence, and region IDs from both representations preserved.
G. Existing non-decomposed VLM pages remain unchanged.
H. Multiple entity types crossing the boundary (TABLE, DIAGRAM, INSTRUCTION, PARAGRAPH, QUESTION).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import types

# Mock sklearn and sentence_transformers
for mod_name in ["sklearn", "sklearn.feature_extraction",
                 "sklearn.feature_extraction.text",
                 "sklearn.metrics", "sklearn.metrics.pairwise",
                 "sentence_transformers"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)
import sklearn.feature_extraction.text as skt
if not hasattr(skt, 'TfidfVectorizer'):
    skt.TfidfVectorizer = type('TfidfVectorizer', (), {'__init__': lambda s, **kw: None})
import sklearn.metrics.pairwise as skp
if not hasattr(skp, 'cosine_similarity'):
    skp.cosine_similarity = lambda x, y: [[0.0]]

from app.models.schemas import (
    DocumentRegion, BBox, Block,
    VLMPageUnderstanding, VLMStructureItem, VLMRelationshipItem,
    DocumentPage, RegionRelationship, DocumentEvidence
)
from app.services.document_understanding_service import DocumentUnderstandingService
from app.services.document_vision_provider import MultimodalDocumentVisionProvider
from app.services.intelligent_question_extraction_service import IntelligentQuestionExtractionService


class TestPostVLMCorruptionFixes:

    def _make_doc_service(self):
        return DocumentUnderstandingService.__new__(DocumentUnderstandingService)

    def _make_vision_provider(self):
        return MultimodalDocumentVisionProvider.__new__(MultimodalDocumentVisionProvider)

    def _make_region(self, region_id: str, text: str, page: int = 1,
                     bbox: BBox = None, region_type: str = "UNKNOWN",
                     confidence: float = 0.5) -> DocumentRegion:
        return DocumentRegion(
            region_id=region_id,
            page=page,
            text=text,
            bbox=bbox or BBox(x=10.0, y=10.0, width=100.0, height=20.0),
            region_type=region_type,
            confidence=confidence,
            evidence=[],
            relationships=[],
            uncertainty=0.0,
            classification_conflict=False,
            conflicting_hypotheses=[],
            verification_state="UNVERIFIED",
            metadata={},
        )

    # 1. Grounded OCR regions are not duplicated as continuation edges
    def test_consumed_ocr_regions_do_not_create_continuation_edges(self):
        svc = self._make_doc_service()

        r1 = self._make_region("w1", "1.", page=1, bbox=BBox(x=10, y=50, width=20, height=20))
        r2 = self._make_region("w2", "Which", page=1, bbox=BBox(x=35, y=50, width=50, height=20))
        r3 = self._make_region("w3", "is", page=1, bbox=BBox(x=90, y=50, width=20, height=20))
        r4 = self._make_region("w4", "correct?", page=1, bbox=BBox(x=115, y=50, width=70, height=20))

        struct = VLMStructureItem(
            role="QUESTION",
            display_number="1",
            confidence=0.98,
            bbox=BBox(x=8, y=48, width=180, height=25),
        )
        vlm_und = VLMPageUnderstanding(
            page_number=1,
            structures=[struct],
            relationships=[],
            semantic_completeness="COMPLETE",
        )

        all_regions = [r1, r2, r3, r4]
        all_relationships = []
        svc._apply_vlm_page_understandings([vlm_und], all_regions, all_relationships, {})

        # Head region gets full consolidated text
        assert r1.region_type == "QUESTION"
        assert r1.text == "1. Which is correct?"

        # Non-head constituent regions are absorbed as UNKNOWN
        assert r2.region_type == "UNKNOWN"
        assert r3.region_type == "UNKNOWN"
        assert r4.region_type == "UNKNOWN"

        # NO continuation_of edges should be created for the consumed constituent blocks
        cont_edges = [r for r in all_relationships if r.relationship_type == "continuation_of"]
        assert len(cont_edges) == 0

    # Requirement A, C, F: Same visual entity crossing decomposition boundary -> merged + metadata/IDs preserved
    def test_req_a_c_f_same_visual_entity_crossing_boundary_merged(self):
        vp = self._make_vision_provider()
        mid_y = 500.0

        q4_top = VLMStructureItem(
            role="QUESTION",
            display_number="4",
            confidence=0.95,
            reasoning="Top part of Q4",
            region_ids=["q4_head_rid"],
            bbox=BBox(x=100, y=460, width=400, height=35),  # y=460..495, near mid_y=500
        )
        top_und = VLMPageUnderstanding(
            page_number=1,
            structures=[q4_top],
            relationships=[],
            finish_reason="STOP",
            semantic_completeness="COMPLETE",
        )

        q4_bot = VLMStructureItem(
            role="QUESTION",
            display_number="4",
            confidence=0.92,
            reasoning="Bottom part of Q4",
            region_ids=["q4_cont_rid"],
            bbox=BBox(x=100, y=0, width=400, height=20),  # in bot crop coords (will be y=500..520)
        )
        opt_a = VLMStructureItem(
            role="OPTION",
            display_label="A",
            confidence=0.95,
            region_ids=["opt_a_id"],
            bbox=BBox(x=120, y=30, width=200, height=20),
        )
        rel_opt = VLMRelationshipItem(
            source_ids=["opt_a_id"],
            target_ids=["q4_cont_rid"],  # Provisional ID from bottom crop
            relationship_type="option_of",
            confidence=0.95,
        )
        bot_und = VLMPageUnderstanding(
            page_number=1,
            structures=[q4_bot, opt_a],
            relationships=[rel_opt],
            finish_reason="STOP",
            semantic_completeness="COMPLETE",
        )

        combined_structures, combined_relationships = vp._reconcile_decomposed_page_understanding(
            top_und=top_und,
            bot_und=bot_und,
            mid_y=mid_y,
        )

        # 1. Exactly one canonical Q4 structure
        q_structs = [s for s in combined_structures if s.role == "QUESTION"]
        assert len(q_structs) == 1
        canon_q = q_structs[0]

        # 2. Region IDs and metadata preserved
        assert "q4_head_rid" in canon_q.region_ids
        assert "q4_cont_rid" in canon_q.region_ids
        assert canon_q.confidence == 0.95
        assert canon_q.bbox.y == 460
        assert canon_q.bbox.height == 60  # 460 to 520

        # 3. Relationship remapped to canonical ID ("q4_head_rid")
        assert len(combined_relationships) == 1
        assert combined_relationships[0].target_ids == ["q4_head_rid"]
        assert combined_relationships[0].source_ids == ["opt_a_id"]

    # Requirement B: Different entities near decomposition boundary -> NOT merged
    def test_req_b_different_entities_near_boundary_not_merged(self):
        vp = self._make_vision_provider()
        mid_y = 500.0

        # Top entity: Q4
        q4_top = VLMStructureItem(
            role="QUESTION",
            display_number="4",
            confidence=0.95,
            region_ids=["q4_id"],
            bbox=BBox(x=100, y=440, width=400, height=30),  # y=440..470
        )
        top_und = VLMPageUnderstanding(page_number=1, structures=[q4_top], relationships=[])

        # Bottom entity: Q5 with different display_number or far below mid_y
        q5_bot = VLMStructureItem(
            role="QUESTION",
            display_number="5",
            confidence=0.95,
            region_ids=["q5_id"],
            bbox=BBox(x=100, y=100, width=400, height=30),  # y in full coords = 600..630 (>100px from mid_y)
        )
        bot_und = VLMPageUnderstanding(page_number=1, structures=[q5_bot], relationships=[])

        combined_structures, _ = vp._reconcile_decomposed_page_understanding(
            top_und=top_und,
            bot_und=bot_und,
            mid_y=mid_y,
        )

        # Both entities must remain distinct
        q_structs = [s for s in combined_structures if s.role == "QUESTION"]
        assert len(q_structs) == 2
        assert {q.display_number for q in q_structs} == {"4", "5"}

    # Requirement D: Duplicate relationships deduplicated
    def test_req_d_duplicate_relationships_deduplicated(self):
        vp = self._make_vision_provider()
        mid_y = 500.0

        q = VLMStructureItem(role="QUESTION", region_ids=["q_id"], bbox=BBox(x=100, y=470, width=400, height=25))
        opt = VLMStructureItem(role="OPTION", region_ids=["opt_id"], bbox=BBox(x=120, y=10, width=200, height=20))

        top_rel = VLMRelationshipItem(source_ids=["opt_id"], target_ids=["q_id"], relationship_type="option_of", confidence=0.95)
        bot_rel = VLMRelationshipItem(source_ids=["opt_id"], target_ids=["q_id"], relationship_type="option_of", confidence=0.95)

        top_und = VLMPageUnderstanding(page_number=1, structures=[q], relationships=[top_rel])
        bot_und = VLMPageUnderstanding(page_number=1, structures=[opt], relationships=[bot_rel])

        _, combined_rels = vp._reconcile_decomposed_page_understanding(top_und, bot_und, mid_y)
        # Duplicate must be deduplicated to exactly 1 relationship
        assert len(combined_rels) == 1

    # Requirement E: Self-loop relationship after remapping -> removed
    def test_req_e_self_loop_relationship_eliminated(self):
        vp = self._make_vision_provider()
        mid_y = 500.0

        # Top and bottom representations of the same entity
        top_s = VLMStructureItem(role="QUESTION", region_ids=["canon_id"], bbox=BBox(x=100, y=480, width=400, height=20))
        bot_s = VLMStructureItem(role="QUESTION", region_ids=["bot_id"], bbox=BBox(x=100, y=0, width=400, height=20))

        # A relationship between bot_id and canon_id that would become a self-loop (canon_id -> canon_id)
        internal_rel = VLMRelationshipItem(source_ids=["bot_id"], target_ids=["canon_id"], relationship_type="continuation_of", confidence=0.9)

        top_und = VLMPageUnderstanding(page_number=1, structures=[top_s], relationships=[])
        bot_und = VLMPageUnderstanding(page_number=1, structures=[bot_s], relationships=[internal_rel])

        _, combined_rels = vp._reconcile_decomposed_page_understanding(top_und, bot_und, mid_y)
        # Self-loop must be dropped
        assert len(combined_rels) == 0

    # Requirement G: Existing non-decomposed VLM pages remain unchanged
    def test_req_g_non_decomposed_page_preserved(self):
        vp = self._make_vision_provider()
        # Normal single VLM call with STOP finish reason (no decomposition triggered)
        und = VLMPageUnderstanding(
            page_number=1,
            structures=[VLMStructureItem(role="QUESTION", region_ids=["q1"], confidence=0.98)],
            relationships=[],
            finish_reason="STOP",
            semantic_completeness="COMPLETE",
        )
        assert und.semantic_completeness == "COMPLETE"
        assert len(und.structures) == 1

    # Requirement H: Multiple entity types crossing the boundary (TABLE, DIAGRAM, INSTRUCTION, PARAGRAPH)
    @pytest.mark.parametrize("role", ["TABLE", "DIAGRAM", "INSTRUCTION", "SECTION_HEADER"])
    def test_req_h_arbitrary_entity_roles_merged_across_boundary(self, role):
        vp = self._make_vision_provider()
        mid_y = 500.0

        top_s = VLMStructureItem(role=role, confidence=0.95, region_ids=[f"{role.lower()}_top"], bbox=BBox(x=50, y=460, width=500, height=35))
        bot_s = VLMStructureItem(role=role, confidence=0.95, region_ids=[f"{role.lower()}_bot"], bbox=BBox(x=50, y=0, width=500, height=40))

        top_und = VLMPageUnderstanding(page_number=1, structures=[top_s], relationships=[])
        bot_und = VLMPageUnderstanding(page_number=1, structures=[bot_s], relationships=[])

        combined_structures, _ = vp._reconcile_decomposed_page_understanding(top_und, bot_und, mid_y)

        # Merged into exactly 1 canonical structure of that role
        role_structs = [s for s in combined_structures if s.role == role]
        assert len(role_structs) == 1
        canon = role_structs[0]
        assert f"{role.lower()}_top" in canon.region_ids
        assert f"{role.lower()}_bot" in canon.region_ids

    # Q1 Clean Extraction Regression
    def test_q1_extraction_no_duplicate_text(self):
        svc = self._make_doc_service()

        r1 = self._make_region("w1", "1.", page=1, bbox=BBox(x=10, y=50, width=20, height=20))
        r2 = self._make_region("w2", "Which", page=1, bbox=BBox(x=35, y=50, width=50, height=20))
        r3 = self._make_region("w3", "of", page=1, bbox=BBox(x=90, y=50, width=20, height=20))
        r4 = self._make_region("w4", "the", page=1, bbox=BBox(x=115, y=50, width=30, height=20))
        r5 = self._make_region("w5", "following?", page=1, bbox=BBox(x=150, y=50, width=70, height=20))

        opt_a = self._make_region("opt_a", "(A) Option 1", page=1, bbox=BBox(x=30, y=80, width=100, height=20))
        opt_b = self._make_region("opt_b", "(B) Option 2", page=1, bbox=BBox(x=30, y=105, width=100, height=20))

        struct_q = VLMStructureItem(role="QUESTION", display_number="1", confidence=0.98, bbox=BBox(x=8, y=48, width=220, height=25))
        struct_a = VLMStructureItem(role="OPTION", display_label="A", confidence=0.95, region_ids=["opt_a"])
        struct_b = VLMStructureItem(role="OPTION", display_label="B", confidence=0.95, region_ids=["opt_b"])

        vlm_und = VLMPageUnderstanding(
            page_number=1,
            structures=[struct_q, struct_a, struct_b],
            relationships=[
                VLMRelationshipItem(source_ids=["opt_a"], target_ids=["w1"], relationship_type="option_of"),
                VLMRelationshipItem(source_ids=["opt_b"], target_ids=["w1"], relationship_type="option_of"),
            ],
            semantic_completeness="COMPLETE",
        )

        all_regions = [r1, r2, r3, r4, r5, opt_a, opt_b]
        all_relationships = []
        svc._apply_vlm_page_understandings([vlm_und], all_regions, all_relationships, {})

        graph = svc._build_structure_graph(
            all_regions=all_regions,
            all_relationships=all_relationships,
            document_purpose="QUESTION_PAPER",
            page_roles={1: "QUESTION_PAPER"},
        )

        extractor = IntelligentQuestionExtractionService(doc_understanding_service=svc)
        from app.models.schemas import DocumentUnderstandingResult
        doc_res = DocumentUnderstandingResult(
            document_id="doc_q1",
            pages=[DocumentPage(page_number=1, width=1000, height=1000, regions=all_regions, reading_order=[r.region_id for r in all_regions])],
            regions=all_regions,
            relationships=all_relationships,
            structure_graph=graph,
            vlm_status="SUCCESS",
        )

        blocks = [Block(id=r.region_id, page=r.page, text=r.text, bbox=r.bbox, confidence=r.confidence) for r in all_regions]
        extraction = extractor.extract_validated_questions(
            blocks=blocks,
            document_id="doc_q1",
            doc_understanding_result=doc_res,
        )

        assert len(extraction.questions) == 1
        q = extraction.questions[0]
        assert q.text == "1. Which of the following?"
        assert len(q.options) == 2

    # Q4 4-Options Full Attachment Regression
    def test_q4_options_preserved_after_cross_boundary_merge(self):
        vp = self._make_vision_provider()
        mid_y = 500.0

        q4_top = VLMStructureItem(
            role="QUESTION",
            display_number="4",
            confidence=0.98,
            region_ids=["q4_head"],
            bbox=BBox(x=100, y=460, width=400, height=35),
        )
        top_und = VLMPageUnderstanding(page_number=1, structures=[q4_top], relationships=[])

        q4_bot = VLMStructureItem(
            role="QUESTION",
            display_number="4",
            confidence=0.95,
            region_ids=["q4_cont"],
            bbox=BBox(x=100, y=0, width=400, height=25),
        )
        opts = [
            VLMStructureItem(role="OPTION", display_label="A", confidence=0.95, region_ids=["opt_a"], bbox=BBox(x=120, y=30, width=150, height=20)),
            VLMStructureItem(role="OPTION", display_label="B", confidence=0.95, region_ids=["opt_b"], bbox=BBox(x=120, y=55, width=150, height=20)),
            VLMStructureItem(role="OPTION", display_label="C", confidence=0.95, region_ids=["opt_c"], bbox=BBox(x=120, y=80, width=150, height=20)),
            VLMStructureItem(role="OPTION", display_label="D", confidence=0.95, region_ids=["opt_d"], bbox=BBox(x=120, y=105, width=150, height=20)),
        ]
        rels = [
            VLMRelationshipItem(source_ids=["opt_a"], target_ids=["q4_cont"], relationship_type="option_of"),
            VLMRelationshipItem(source_ids=["opt_b"], target_ids=["q4_cont"], relationship_type="option_of"),
            VLMRelationshipItem(source_ids=["opt_c"], target_ids=["q4_cont"], relationship_type="option_of"),
            VLMRelationshipItem(source_ids=["opt_d"], target_ids=["q4_cont"], relationship_type="option_of"),
        ]
        bot_und = VLMPageUnderstanding(page_number=1, structures=[q4_bot] + opts, relationships=rels)

        comb_structs, comb_rels = vp._reconcile_decomposed_page_understanding(top_und, bot_und, mid_y)

        # Exactly 1 canonical question
        q_structs = [s for s in comb_structs if s.role == "QUESTION"]
        assert len(q_structs) == 1

        # Verify all 4 relationships point to "q4_head"
        assert len(comb_rels) == 4
        for rel in comb_rels:
            assert rel.target_ids == ["q4_head"]


    # ================================================================
    # Test-I: cont_ids must never absorb another structure head ID
    # ================================================================

    def test_req_i_cont_ids_never_absorb_another_structures_head(self):
        svc = DocumentUnderstandingService.__new__(DocumentUnderstandingService)
        q1_struct = VLMStructureItem(role='QUESTION', confidence=0.95,
            region_ids=['q1_head', 'q2_head'], bbox=BBox(x=10, y=10, width=500, height=30))
        q2_struct = VLMStructureItem(role='QUESTION', confidence=0.95,
            region_ids=['q2_head'], bbox=BBox(x=10, y=50, width=500, height=30))
        regions = [
            self._make_region('q1_head', 'Q1. Photosynthesis?', bbox=BBox(x=10, y=10, width=500, height=30)),
            self._make_region('q2_head', 'Q2. Define osmosis.', bbox=BBox(x=10, y=50, width=500, height=30)),
        ]
        und = VLMPageUnderstanding(page_number=1, structures=[q1_struct, q2_struct],
            relationships=[], semantic_completeness='COMPLETE', finish_reason='STOP')
        all_rels = []
        svc._apply_vlm_page_understandings(vlm_understandings=[und], all_regions=regions, all_relationships=all_rels, pages_dict={})
        q2_r = next(r for r in regions if r.region_id == 'q2_head')
        assert q2_r.region_type == 'QUESTION', 'Q2 head was absorbed by Q1 cont_ids'

    # ================================================================
    # Test-J: _attach_options accepts UNKNOWN-role nodes with option_of edge
    # ================================================================

    def test_req_j_attach_options_accepts_unknown_role_option_nodes(self):
        from app.models.schemas import GraphNode, GraphEdge, Question
        svc = IntelligentQuestionExtractionService.__new__(IntelligentQuestionExtractionService)
        q_node = GraphNode(region_id='q1', role='QUESTION', text='Q1. Which element?',
                           page=1, bbox=BBox(x=10, y=10, width=400, height=25), confidence=0.95)
        opt_node = GraphNode(region_id='opt_a', role='UNKNOWN', text='A. Hydrogen',
                             page=1, bbox=BBox(x=10, y=40, width=200, height=20), confidence=0.90)
        nodes = {'q1': q_node, 'opt_a': opt_node}
        region_map = {
            'q1': self._make_region('q1', 'Q1. Which element?', region_type='QUESTION',
                                    bbox=BBox(x=10, y=10, width=400, height=25), confidence=0.95),
            'opt_a': self._make_region('opt_a', 'A. Hydrogen', region_type='UNKNOWN',
                                       bbox=BBox(x=10, y=40, width=200, height=20), confidence=0.90),
        }
        q_obj = Question(id='Q1', number='1', text='Q1. Which element?',
                         page=1, bbox=BBox(x=10, y=10, width=400, height=25))
        children_of = {'q1': [('opt_a', 'option_of', 0.90)]}
        count = svc._attach_options(question=q_obj, question_node_id='q1', children_of=children_of,
            graph_nodes=nodes, region_map=region_map, page_regions_by_page={1: list(region_map.values())})
        assert count == 1, f'Expected 1 option, got {count}'
        assert 'Hydrogen' in q_obj.extracted_options[0].text

    # ================================================================
    # Test-K: question type deferred to after option attachment
    # ================================================================

    def test_req_k_question_type_set_after_option_attachment(self):
        from app.models.schemas import GraphNode, GraphEdge, DocumentStructureGraph, DocumentUnderstandingResult, DocumentPage
        svc = IntelligentQuestionExtractionService.__new__(IntelligentQuestionExtractionService)

        def _gr(q_id, q_text, opts):
            nodes = {q_id: GraphNode(region_id=q_id, role='QUESTION', text=q_text,
                                     page=1, bbox=BBox(x=10, y=10, width=500, height=30), confidence=0.95)}
            edges = []
            for i, (oid, otxt) in enumerate(opts):
                nodes[oid] = GraphNode(region_id=oid, role='OPTION', text=otxt,
                                       page=1, bbox=BBox(x=10, y=50+i*25, width=400, height=20), confidence=0.9)
                edges.append(GraphEdge(source_id=oid, target_id=q_id, relationship='option_of',
                                       confidence=0.9, semantic_state='CONFIDENT'))
            return DocumentStructureGraph(nodes=nodes, edges=edges)

        def _doc(regs, graph):
            return DocumentUnderstandingResult(document_id='t', regions=regs,
                pages=[DocumentPage(page_number=1, width=600.0, height=800.0)],
                structure_graph=graph, relationships=[])

        g1 = _gr('q1', 'Q1. Which?', [('a', 'A. Alpha'), ('b', 'B. Beta')])
        r1 = [self._make_region('q1', 'Q1. Which?', region_type='QUESTION',
                                bbox=BBox(x=10, y=10, width=500, height=30), confidence=0.95),
              self._make_region('a', 'A. Alpha', region_type='OPTION',
                                bbox=BBox(x=10, y=50, width=400, height=20), confidence=0.9),
              self._make_region('b', 'B. Beta', region_type='OPTION',
                                bbox=BBox(x=10, y=75, width=400, height=20), confidence=0.9)]
        res1 = svc._extract_from_graph(g1, _doc(r1, g1), 't1')
        assert res1.questions[0].question_type == 'MCQ'

        long_text = 'Q2. ' + 'Describe oxidation in detail with examples and equations. ' * 5
        g2 = _gr('q2', long_text, [])
        r2 = [self._make_region('q2', long_text, region_type='QUESTION',
                                bbox=BBox(x=10, y=10, width=500, height=30), confidence=0.95)]
        res2 = svc._extract_from_graph(g2, _doc(r2, g2), 't2')
        assert res2.questions[0].question_type == 'LONG_ANSWER'

        g3 = _gr('q3', 'Q3. Define osmosis.', [])
        r3 = [self._make_region('q3', 'Q3. Define osmosis.', region_type='QUESTION',
                                bbox=BBox(x=10, y=10, width=500, height=30), confidence=0.95)]
        res3 = svc._extract_from_graph(g3, _doc(r3, g3), 't3')
        assert res3.questions[0].question_type == 'SHORT_ANSWER'

    # ================================================================
    # Test-L: Multi-question page: each question gets only its own options
    # ================================================================

    def test_req_l_multi_question_page_options_isolated(self):
        from collections import defaultdict
        svc = DocumentUnderstandingService.__new__(DocumentUnderstandingService)
        q1 = VLMStructureItem(role='QUESTION', confidence=0.95,
                              region_ids=['q1h'], bbox=BBox(x=10, y=10, width=500, height=30))
        q2 = VLMStructureItem(role='QUESTION', confidence=0.95,
                              region_ids=['q2h'], bbox=BBox(x=10, y=200, width=500, height=30))
        o1a = VLMStructureItem(role='OPTION', confidence=0.95, region_ids=['o1a'], bbox=BBox(x=10, y=50, width=400, height=20))
        o1b = VLMStructureItem(role='OPTION', confidence=0.95, region_ids=['o1b'], bbox=BBox(x=10, y=75, width=400, height=20))
        o2a = VLMStructureItem(role='OPTION', confidence=0.95, region_ids=['o2a'], bbox=BBox(x=10, y=240, width=400, height=20))
        o2b = VLMStructureItem(role='OPTION', confidence=0.95, region_ids=['o2b'], bbox=BBox(x=10, y=265, width=400, height=20))
        rels = [
            VLMRelationshipItem(source_ids=['o1a'], target_ids=['q1h'], relationship_type='option_of', confidence=0.95),
            VLMRelationshipItem(source_ids=['o1b'], target_ids=['q1h'], relationship_type='option_of', confidence=0.95),
            VLMRelationshipItem(source_ids=['o2a'], target_ids=['q2h'], relationship_type='option_of', confidence=0.95),
            VLMRelationshipItem(source_ids=['o2b'], target_ids=['q2h'], relationship_type='option_of', confidence=0.95),
        ]
        und = VLMPageUnderstanding(page_number=1, structures=[q1, q2, o1a, o1b, o2a, o2b],
            relationships=rels, semantic_completeness='COMPLETE', finish_reason='STOP')
        regions = [
            self._make_region('q1h', 'Q1. 2+2?', bbox=BBox(x=10, y=10, width=500, height=30), confidence=0.95),
            self._make_region('q2h', 'Q2. 3+3?', bbox=BBox(x=10, y=200, width=500, height=30), confidence=0.95),
            self._make_region('o1a', 'A. 3', bbox=BBox(x=10, y=50, width=400, height=20), confidence=0.90),
            self._make_region('o1b', 'B. 4', bbox=BBox(x=10, y=75, width=400, height=20), confidence=0.90),
            self._make_region('o2a', 'A. 5', bbox=BBox(x=10, y=240, width=400, height=20), confidence=0.90),
            self._make_region('o2b', 'B. 6', bbox=BBox(x=10, y=265, width=400, height=20), confidence=0.90),
        ]
        all_rels = []
        svc._apply_vlm_page_understandings(vlm_understandings=[und], all_regions=regions, all_relationships=all_rels, pages_dict={})
        by_q = defaultdict(list)
        for r in all_rels:
            if r.relationship_type == 'option_of':
                by_q[r.target_region_id].append(r.source_region_id)
        assert set(by_q.get('q1h', [])) == {'o1a', 'o1b'}, f'Q1 wrong: {by_q.get(chr(113)+chr(49)+chr(104))}'
        assert set(by_q.get('q2h', [])) == {'o2a', 'o2b'}, f'Q2 wrong: {by_q.get(chr(113)+chr(50)+chr(104))}'
