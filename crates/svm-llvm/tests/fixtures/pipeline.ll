; ModuleID = '/tmp/claude-0/-home-user-vm/8192c67a-39d3-5f8d-8db5-366ce6cb9abb/scratchpad/pipeline.bc'
source_filename = "pipeline.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@.str = private unnamed_addr constant [4 x i8] c"gen\00", align 1
@.str.1 = private unnamed_addr constant [3 x i8] c"up\00", align 1
@__px_handle = internal unnamed_addr global i32 -1, align 4
@.str.3 = private unnamed_addr constant [6 x i8] c"posix\00", align 1
@str = private unnamed_addr constant [12 x i8] c"pipeline ok\00", align 1

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca [2 x i32], align 4
  %2 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %1) #4
  %3 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %4 = icmp slt i32 %3, 0
  br i1 %4, label %5, label %7

5:                                                ; preds = %0
  %6 = tail call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #4
  store i32 %6, ptr @__px_handle, align 4, !tbaa !5
  br label %7

7:                                                ; preds = %0, %5
  %8 = phi i32 [ %6, %5 ], [ %3, %0 ]
  %9 = ptrtoint ptr %1 to i64
  %10 = call i64 @__vm_host_call(i32 noundef %8, i32 noundef 23, i64 noundef %9, i64 noundef 0, i64 noundef 0, i64 noundef 0) #4
  %11 = and i64 %10, 4294967295
  %12 = icmp eq i64 %11, 0
  br i1 %12, label %13, label %116

13:                                               ; preds = %7
  %14 = load i32, ptr %1, align 4, !tbaa !5
  %15 = getelementptr inbounds [2 x i32], ptr %1, i64 0, i64 1
  %16 = load i32, ptr %15, align 4, !tbaa !5
  %17 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %18 = icmp slt i32 %17, 0
  br i1 %18, label %19, label %21

19:                                               ; preds = %13
  %20 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #4
  store i32 %20, ptr @__px_handle, align 4, !tbaa !5
  br label %21

21:                                               ; preds = %13, %19
  %22 = phi i32 [ %20, %19 ], [ %17, %13 ]
  %23 = call i64 @__vm_host_call(i32 noundef %22, i32 noundef 25, i64 noundef 1, i64 noundef 0, i64 noundef 0, i64 noundef 0) #4
  %24 = and i64 %23, 2147483648
  %25 = icmp eq i64 %24, 0
  br i1 %25, label %26, label %116

26:                                               ; preds = %21
  %27 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %28 = icmp slt i32 %27, 0
  br i1 %28, label %29, label %31

29:                                               ; preds = %26
  %30 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #4
  store i32 %30, ptr @__px_handle, align 4, !tbaa !5
  br label %31

31:                                               ; preds = %26, %29
  %32 = phi i32 [ %30, %29 ], [ %27, %26 ]
  %33 = sext i32 %16 to i64
  %34 = call i64 @__vm_host_call(i32 noundef %32, i32 noundef 24, i64 noundef %33, i64 noundef 1, i64 noundef 0, i64 noundef 0) #4
  %35 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %36 = icmp slt i32 %35, 0
  br i1 %36, label %37, label %39

37:                                               ; preds = %31
  %38 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #4
  store i32 %38, ptr @__px_handle, align 4, !tbaa !5
  br label %39

39:                                               ; preds = %31, %37
  %40 = phi i32 [ %38, %37 ], [ %35, %31 ]
  %41 = call i64 @__vm_host_call(i32 noundef %40, i32 noundef 6, i64 noundef %33, i64 noundef 0, i64 noundef 0, i64 noundef 0) #4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %2) #4
  store i32 0, ptr %2, align 4, !tbaa !5
  %42 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %43 = icmp slt i32 %42, 0
  br i1 %43, label %44, label %46

44:                                               ; preds = %39
  %45 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #4
  store i32 %45, ptr @__px_handle, align 4, !tbaa !5
  br label %46

46:                                               ; preds = %39, %44
  %47 = phi i32 [ %45, %44 ], [ %42, %39 ]
  %48 = call i64 @__vm_host_call(i32 noundef %47, i32 noundef 27, i64 noundef ptrtoint (ptr @.str to i64), i64 noundef 3, i64 noundef 0, i64 noundef 0) #4
  %49 = icmp slt i64 %48, 0
  br i1 %49, label %114, label %50

50:                                               ; preds = %46
  %51 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %52 = icmp slt i32 %51, 0
  br i1 %52, label %53, label %55

53:                                               ; preds = %50
  %54 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #4
  store i32 %54, ptr @__px_handle, align 4, !tbaa !5
  br label %55

55:                                               ; preds = %50, %53
  %56 = phi i32 [ %54, %53 ], [ %51, %50 ]
  %57 = ptrtoint ptr %2 to i64
  %58 = call i64 @__vm_host_call(i32 noundef %56, i32 noundef 28, i64 noundef %48, i64 noundef %57, i64 noundef 0, i64 noundef 0) #4
  %59 = trunc i64 %58 to i32
  %60 = trunc i64 %48 to i32
  %61 = icmp eq i32 %59, %60
  br i1 %61, label %62, label %114

62:                                               ; preds = %55
  %63 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %64 = icmp slt i32 %63, 0
  br i1 %64, label %65, label %67

65:                                               ; preds = %62
  %66 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #4
  store i32 %66, ptr @__px_handle, align 4, !tbaa !5
  br label %67

67:                                               ; preds = %62, %65
  %68 = phi i32 [ %66, %65 ], [ %63, %62 ]
  %69 = and i64 %23, 2147483647
  %70 = call i64 @__vm_host_call(i32 noundef %68, i32 noundef 24, i64 noundef %69, i64 noundef 1, i64 noundef 0, i64 noundef 0) #4
  %71 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %72 = icmp slt i32 %71, 0
  br i1 %72, label %73, label %75

73:                                               ; preds = %67
  %74 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #4
  store i32 %74, ptr @__px_handle, align 4, !tbaa !5
  br label %75

75:                                               ; preds = %67, %73
  %76 = phi i32 [ %74, %73 ], [ %71, %67 ]
  %77 = call i64 @__vm_host_call(i32 noundef %76, i32 noundef 6, i64 noundef %69, i64 noundef 0, i64 noundef 0, i64 noundef 0) #4
  %78 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %79 = icmp slt i32 %78, 0
  br i1 %79, label %80, label %82

80:                                               ; preds = %75
  %81 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #4
  store i32 %81, ptr @__px_handle, align 4, !tbaa !5
  br label %82

82:                                               ; preds = %75, %80
  %83 = phi i32 [ %81, %80 ], [ %78, %75 ]
  %84 = sext i32 %14 to i64
  %85 = call i64 @__vm_host_call(i32 noundef %83, i32 noundef 24, i64 noundef %84, i64 noundef 0, i64 noundef 0, i64 noundef 0) #4
  %86 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %87 = icmp slt i32 %86, 0
  br i1 %87, label %88, label %90

88:                                               ; preds = %82
  %89 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #4
  store i32 %89, ptr @__px_handle, align 4, !tbaa !5
  br label %90

90:                                               ; preds = %82, %88
  %91 = phi i32 [ %89, %88 ], [ %86, %82 ]
  %92 = call i64 @__vm_host_call(i32 noundef %91, i32 noundef 6, i64 noundef %84, i64 noundef 0, i64 noundef 0, i64 noundef 0) #4
  %93 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %94 = icmp slt i32 %93, 0
  br i1 %94, label %95, label %97

95:                                               ; preds = %90
  %96 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #4
  store i32 %96, ptr @__px_handle, align 4, !tbaa !5
  br label %97

97:                                               ; preds = %90, %95
  %98 = phi i32 [ %96, %95 ], [ %93, %90 ]
  %99 = call i64 @__vm_host_call(i32 noundef %98, i32 noundef 27, i64 noundef ptrtoint (ptr @.str.1 to i64), i64 noundef 2, i64 noundef 0, i64 noundef 0) #4
  %100 = icmp slt i64 %99, 0
  br i1 %100, label %114, label %101

101:                                              ; preds = %97
  %102 = load i32, ptr @__px_handle, align 4, !tbaa !5
  %103 = icmp slt i32 %102, 0
  br i1 %103, label %104, label %106

104:                                              ; preds = %101
  %105 = call i32 @__vm_cap_resolve(ptr noundef nonnull @.str.3, i64 noundef 5) #4
  store i32 %105, ptr @__px_handle, align 4, !tbaa !5
  br label %106

106:                                              ; preds = %101, %104
  %107 = phi i32 [ %105, %104 ], [ %102, %101 ]
  %108 = call i64 @__vm_host_call(i32 noundef %107, i32 noundef 28, i64 noundef %99, i64 noundef %57, i64 noundef 0, i64 noundef 0) #4
  %109 = trunc i64 %108 to i32
  %110 = trunc i64 %99 to i32
  %111 = icmp eq i32 %109, %110
  br i1 %111, label %112, label %114

112:                                              ; preds = %106
  %113 = call i32 @puts(ptr nonnull dereferenceable(1) @str)
  br label %114

114:                                              ; preds = %112, %97, %106, %55, %46
  %115 = phi i32 [ 3, %46 ], [ 4, %55 ], [ 0, %112 ], [ 5, %97 ], [ 6, %106 ]
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %2) #4
  br label %116

116:                                              ; preds = %114, %21, %7
  %117 = phi i32 [ 1, %7 ], [ %115, %114 ], [ 2, %21 ]
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %1) #4
  ret i32 %117
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

declare i64 @__vm_host_call(i32 noundef, i32 noundef, i64 noundef, i64 noundef, i64 noundef, i64 noundef) local_unnamed_addr #2

declare i32 @__vm_cap_resolve(ptr noundef, i64 noundef) local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #3

attributes #0 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nofree nounwind }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Ubuntu clang version 18.1.3 (1ubuntu1)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
