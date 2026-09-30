; ModuleID = 'posix_probe.bc'
source_filename = "posix_probe.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@.str = private unnamed_addr constant [6 x i8] c"posix\00", align 1
@px = internal unnamed_addr global i32 0, align 4
@.str.1 = private unnamed_addr constant [5 x i8] c"ping\00", align 1
@.str.2 = private unnamed_addr constant [8 x i8] c"/bin/up\00", align 1
@str = private unnamed_addr constant [15 x i8] c"posix probe ok\00", align 1

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca [2 x i32], align 4
  %2 = alloca [8 x i8], align 4
  %3 = alloca [5 x i64], align 16
  %4 = alloca i32, align 4
  %5 = tail call i32 @__vm_cap_resolve(ptr noundef nonnull @.str, i64 noundef 5) #5
  store i32 %5, ptr @px, align 4, !tbaa !5
  %6 = icmp slt i32 %5, 0
  br i1 %6, label %58, label %7

7:                                                ; preds = %0
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %1) #5
  %8 = ptrtoint ptr %1 to i64
  %9 = call i64 @__vm_host_call(i32 noundef %5, i32 noundef 23, i64 noundef %8, i64 noundef 0, i64 noundef 0, i64 noundef 0) #5
  %10 = icmp eq i64 %9, 0
  br i1 %10, label %11, label %56

11:                                               ; preds = %7
  %12 = load i32, ptr %1, align 4, !tbaa !5
  %13 = getelementptr inbounds [2 x i32], ptr %1, i64 0, i64 1
  %14 = load i32, ptr %13, align 4, !tbaa !5
  %15 = sext i32 %14 to i64
  %16 = load i32, ptr @px, align 4, !tbaa !5
  %17 = call i64 @__vm_host_call(i32 noundef %16, i32 noundef 24, i64 noundef %15, i64 noundef 9, i64 noundef 0, i64 noundef 0) #5
  %18 = icmp eq i64 %17, 9
  br i1 %18, label %19, label %56

19:                                               ; preds = %11
  %20 = load i32, ptr @px, align 4, !tbaa !5
  %21 = call i64 @__vm_host_call(i32 noundef %20, i32 noundef 0, i64 noundef 9, i64 noundef ptrtoint (ptr @.str.1 to i64), i64 noundef 4, i64 noundef 0) #5
  %22 = icmp eq i64 %21, 4
  br i1 %22, label %23, label %56

23:                                               ; preds = %19
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %2) #5
  %24 = sext i32 %12 to i64
  %25 = ptrtoint ptr %2 to i64
  %26 = load i32, ptr @px, align 4, !tbaa !5
  %27 = call i64 @__vm_host_call(i32 noundef %26, i32 noundef 1, i64 noundef %24, i64 noundef %25, i64 noundef 8, i64 noundef 0) #5
  %28 = icmp eq i64 %27, 4
  br i1 %28, label %29, label %54

29:                                               ; preds = %23
  %30 = load <4 x i8>, ptr %2, align 4
  %31 = freeze <4 x i8> %30
  %32 = bitcast <4 x i8> %31 to i32
  %33 = icmp eq i32 %32, 1735289200
  br i1 %33, label %34, label %54

34:                                               ; preds = %29
  call void @llvm.lifetime.start.p0(i64 40, ptr nonnull %3) #5
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(40) %3, i8 0, i64 40, i1 false)
  store i64 ptrtoint (ptr @.str.2 to i64), ptr %3, align 16
  %35 = ptrtoint ptr %3 to i64
  %36 = load i32, ptr @px, align 4, !tbaa !5
  %37 = call i64 @__vm_host_call(i32 noundef %36, i32 noundef 62, i64 noundef %35, i64 noundef 0, i64 noundef 0, i64 noundef 0) #5
  %38 = icmp slt i64 %37, 0
  br i1 %38, label %52, label %39

39:                                               ; preds = %34
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %4) #5
  store i32 0, ptr %4, align 4, !tbaa !5
  %40 = ptrtoint ptr %4 to i64
  %41 = load i32, ptr @px, align 4, !tbaa !5
  %42 = call i64 @__vm_host_call(i32 noundef %41, i32 noundef 28, i64 noundef %37, i64 noundef %40, i64 noundef 0, i64 noundef 0) #5
  %43 = icmp eq i64 %42, %37
  br i1 %43, label %44, label %50

44:                                               ; preds = %39
  %45 = load i32, ptr %4, align 4, !tbaa !5
  %46 = and i32 %45, 65280
  %47 = icmp eq i32 %46, 10752
  br i1 %47, label %48, label %50

48:                                               ; preds = %44
  %49 = call i32 @puts(ptr nonnull dereferenceable(1) @str)
  br label %50

50:                                               ; preds = %44, %39, %48
  %51 = phi i32 [ 0, %48 ], [ 8, %39 ], [ 9, %44 ]
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %4) #5
  br label %52

52:                                               ; preds = %34, %50
  %53 = phi i32 [ %51, %50 ], [ 7, %34 ]
  call void @llvm.lifetime.end.p0(i64 40, ptr nonnull %3) #5
  br label %54

54:                                               ; preds = %29, %23, %52
  %55 = phi i32 [ %53, %52 ], [ 5, %23 ], [ 6, %29 ]
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %2) #5
  br label %56

56:                                               ; preds = %54, %11, %19, %7
  %57 = phi i32 [ 2, %7 ], [ %55, %54 ], [ 3, %11 ], [ 4, %19 ]
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %1) #5
  br label %58

58:                                               ; preds = %0, %56
  %59 = phi i32 [ %57, %56 ], [ 1, %0 ]
  ret i32 %59
}

declare i32 @__vm_cap_resolve(ptr noundef, i64 noundef) local_unnamed_addr #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

declare i64 @__vm_host_call(i32 noundef, i32 noundef, i64 noundef, i64 noundef, i64 noundef, i64 noundef) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #4

attributes #0 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #4 = { nofree nounwind }
attributes #5 = { nounwind }

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
