"""
Database schema for academic papers on dog behavior.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import json


# class Author(BaseModel):
#     """Author information."""
#     name: str
#     affiliation: Optional[str] = None
#     email: Optional[str] = None

# class Content(BaseModel):
#     """Content of the paper."""
#     abstract: str = Field(..., description="Abstract of the paper")
#     body: str = Field(..., description="Main part of the paper if or Full text content of the paper if the other fields are not available")
#     conclusion: str = Field(..., description="Conclusion of the paper")
#     references: str = Field(..., description="References cited in the paper")

class Paper(BaseModel):
    """Academic paper model."""
    id: str = Field(..., description="Unique paper identifier")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    abstract: str = Field(..., description="Abstract of the paper")
    body: str = Field(..., description="Main content of the paper")
    conclusion: str = Field(..., description="Conclusion of the paper")
    references: str = Field(..., description="References cited in the paper")
    keywords: List[str] = Field(default_factory=list, description="Paper keywords")
    subject: Optional[str] = None
    publication_date: Optional[datetime] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    # url: Optional[str] = None
    
    # Dog behavior specific fields
    dog_breeds: List[str] = Field(default_factory=list, description="Dog breeds studied")
    behavior_categories: List[str] = Field(default_factory=list, description="Behavior categories")
    study_type: Optional[str] = None  # experimental, observational, review, etc.
    # sample_size: Optional[int] = None
    # age_range: Optional[str] = None  # puppy, adult, senior, etc.
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Full metadata of the paper")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for vector database storage."""
        return {
            "id": self.id,
            "title": self.title if self.title else "",
            "authors": json.dumps([author for author in self.authors]),
            "subject": self.subject if self.subject else "",
            "abstract": self.abstract if self.abstract else "",
            "body": self.body if self.body else "",
            "conclusion": self.conclusion if self.conclusion else "",
            "references": self.references if self.references else "",
            "keywords": json.dumps(self.keywords) if self.keywords else "",
            "publication_date": self.publication_date.isoformat() if self.publication_date else "",
            "journal": self.journal if self.journal else "",
            "doi": self.doi if self.doi else "",
            # "url": self.url if self.url else "",
            "dog_breeds": json.dumps(self.dog_breeds),
            "behavior_categories": json.dumps(self.behavior_categories),
            "study_type": self.study_type if self.study_type else "",
            # "sample_size": self.sample_size,
            # "age_range": self.age_range,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else "",
        }


class BehaviorCategories:
    """Common dog behavior categories for classification."""
    
    AGGRESSION = "aggression"
    ANXIETY = "anxiety"
    SEPARATION_ANXIETY = "separation_anxiety"
    FEAR = "fear"
    SOCIALIZATION = "socialization"
    TRAINING = "training"
    PLAY_BEHAVIOR = "play_behavior"
    COMMUNICATION = "communication"
    TERRITORIAL = "territorial"
    DESTRUCTIVE = "destructive"
    BARKING = "barking"
    RESOURCE_GUARDING = "resource_guarding"
    LEARNING = "learning"
    COGNITION = "cognition"
    STRESS = "stress"
    DOMINANCE = "dominance"
    PACK_BEHAVIOR = "pack_behavior"
    HUNTING = "hunting"
    FEEDING = "feeding"
    SLEEPING = "sleeping"
    GROOMING = "grooming"
    MATERNAL = "maternal"
    SEXUAL = "sexual"
    PAIN_RELATED = "pain_related"
    MEDICATION_EFFECTS = "medication_effects"
    
    @classmethod
    def get_all_categories(cls) -> List[str]:
        """Get all behavior categories."""
        return [
            cls.AGGRESSION, cls.ANXIETY, cls.SEPARATION_ANXIETY, cls.FEAR,
            cls.SOCIALIZATION, cls.TRAINING, cls.PLAY_BEHAVIOR, cls.COMMUNICATION,
            cls.TERRITORIAL, cls.DESTRUCTIVE, cls.BARKING, cls.RESOURCE_GUARDING,
            cls.LEARNING, cls.COGNITION, cls.STRESS, cls.DOMINANCE,
            cls.PACK_BEHAVIOR, cls.HUNTING, cls.FEEDING, cls.SLEEPING,
            cls.GROOMING, cls.MATERNAL, cls.SEXUAL, cls.PAIN_RELATED,
            cls.MEDICATION_EFFECTS
        ]


class CommonDogBreeds:
    """Common dog breeds for classification."""
    
    # Working breeds
    GERMAN_SHEPHERD = "german_shepherd"
    ROTTWEILER = "rottweiler"
    DOBERMAN = "doberman"
    BOXER = "boxer"
    SIBERIAN_HUSKY = "siberian_husky"
    
    # Sporting breeds
    LABRADOR = "labrador"
    GOLDEN_RETRIEVER = "golden_retriever"
    POINTER = "pointer"
    SETTER = "setter"
    SPANIEL = "spaniel"
    
    # Terrier breeds
    BULL_TERRIER = "bull_terrier"
    JACK_RUSSELL = "jack_russell"
    YORKSHIRE_TERRIER = "yorkshire_terrier"
    
    # Toy breeds
    CHIHUAHUA = "chihuahua"
    PUG = "pug"
    MALTESE = "maltese"
    
    # Herding breeds
    BORDER_COLLIE = "border_collie"
    AUSTRALIAN_SHEPHERD = "australian_shepherd"
    CORGI = "corgi"
    
    # Hound breeds
    BEAGLE = "beagle"
    BLOODHOUND = "bloodhound"
    GREYHOUND = "greyhound"
    
    # Other common breeds
    BULLDOG = "bulldog"
    POODLE = "poodle"
    MIXED_BREED = "mixed_breed"
    
    @classmethod
    def get_all_breeds(cls) -> List[str]:
        """Get all dog breeds."""
        return [
            cls.GERMAN_SHEPHERD, cls.ROTTWEILER, cls.DOBERMAN, cls.BOXER,
            cls.SIBERIAN_HUSKY, cls.LABRADOR, cls.GOLDEN_RETRIEVER,
            cls.POINTER, cls.SETTER, cls.SPANIEL, cls.BULL_TERRIER,
            cls.JACK_RUSSELL, cls.YORKSHIRE_TERRIER, cls.CHIHUAHUA,
            cls.PUG, cls.MALTESE, cls.BORDER_COLLIE, cls.AUSTRALIAN_SHEPHERD,
            cls.CORGI, cls.BEAGLE, cls.BLOODHOUND, cls.GREYHOUND,
            cls.BULLDOG, cls.POODLE, cls.MIXED_BREED
        ]
