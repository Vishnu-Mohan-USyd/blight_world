#include "BlightWorldPrimitive.h"

#include "Components/SceneComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "UObject/ConstructorHelpers.h"

ABlightWorldPrimitive::ABlightWorldPrimitive()
{
    PrimaryActorTick.bCanEverTick = false;
    SetRootComponent(CreateDefaultSubobject<USceneComponent>(TEXT("Root")));
    PreviewMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("PreviewMesh"));
    PreviewMesh->SetupAttachment(GetRootComponent());

    static ConstructorHelpers::FObjectFinder<UStaticMesh> BoxAsset(TEXT("/Engine/BasicShapes/Cube.Cube"));
    static ConstructorHelpers::FObjectFinder<UStaticMesh> SphereAsset(TEXT("/Engine/BasicShapes/Sphere.Sphere"));
    static ConstructorHelpers::FObjectFinder<UStaticMesh> CylinderAsset(TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
    static ConstructorHelpers::FObjectFinder<UStaticMesh> PlaneAsset(TEXT("/Engine/BasicShapes/Plane.Plane"));
    BoxMesh = BoxAsset.Object;
    SphereMesh = SphereAsset.Object;
    CylinderMesh = CylinderAsset.Object;
    PlaneMesh = PlaneAsset.Object;
    UpdatePreview();
}

void ABlightWorldPrimitive::OnConstruction(const FTransform& Transform)
{
    Super::OnConstruction(Transform);
    UpdatePreview();
}

void ABlightWorldPrimitive::UpdatePreview()
{
    UStaticMesh* Mesh = nullptr;
    FVector SizeM = FVector::ZeroVector;
    switch (Shape)
    {
    case EBlightWorldPrimitiveShape::Box:
        Mesh = BoxMesh;
        SizeM = 2.0 * HalfSizeM;
        break;
    case EBlightWorldPrimitiveShape::Sphere:
        Mesh = SphereMesh;
        SizeM = FVector(2.0 * RadiusM);
        break;
    case EBlightWorldPrimitiveShape::Cylinder:
        Mesh = CylinderMesh;
        SizeM = FVector(2.0 * RadiusM, 2.0 * RadiusM, 2.0 * HalfHeightM);
        break;
    case EBlightWorldPrimitiveShape::Plane:
        Mesh = PlaneMesh;
        SizeM = FVector(2.0 * HalfSizeM.X, 2.0 * HalfSizeM.Y, 0.0);
        break;
    }

    PreviewMesh->SetSimulatePhysics(false);
    PreviewMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    PreviewMesh->SetGenerateOverlapEvents(false);
    PreviewMesh->SetStaticMesh(Mesh);
    if (Mesh)
    {
        const FBox Bounds = Mesh->GetBoundingBox();
        const FVector NativeSizeCm = Bounds.GetSize();
        FVector Scale = FVector::OneVector;
        for (int32 Axis = 0; Axis < 3; ++Axis)
        {
            if (NativeSizeCm[Axis] > UE_SMALL_NUMBER)
            {
                Scale[Axis] = 100.0 * SizeM[Axis] / NativeSizeCm[Axis];
            }
        }
        PreviewMesh->SetRelativeScale3D(Scale);
        PreviewMesh->SetRelativeLocation(-Bounds.GetCenter() * Scale);
    }
}
